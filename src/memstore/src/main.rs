use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use hnsw_rs::anndists::dist::distances::DistCosine;
use hnsw_rs::prelude::{Hnsw, Neighbour};

#[derive(Clone, Debug)]
struct Record {
    id: u128,
    ts: i64,
    kind: String,
    weight: f32,
    text: String,
}

const VECTOR_DIM: usize = 256;
const HNSW_M: usize = 16;
const HNSW_EF_CONSTRUCTION: usize = 200;
const HNSW_NB_LAYER: usize = 16;
const HNSW_EF_SEARCH: usize = 50;

fn main() {
    let mut args = env::args().skip(1);
    let Some(cmd) = args.next() else {
        print_usage();
        return;
    };

    let rest: Vec<String> = args.collect();
    let result = match cmd.as_str() {
        "add" => cmd_add(&rest),
        "search" => cmd_search(&rest),
        "recent" => cmd_recent(&rest),
        "compact" => cmd_compact(&rest),
        "help" | "-h" | "--help" => {
            print_usage();
            Ok(())
        }
        _ => {
            eprintln!("Unknown command: {cmd}");
            print_usage();
            Err("unknown command")
        }
    };

    if result.is_err() {
        std::process::exit(1);
    }
}

fn print_usage() {
    eprintln!(
        "memstore - simple local memory store\n\n")
    ;
    eprintln!("Commands:");
    eprintln!("  add     --text <text> [--kind <kind>] [--weight <w>] [--path <file>] [--vec-path <file>]");
    eprintln!("  search  --query <text> [--limit <n>] [--path <file>] [--vec-path <file>]");
    eprintln!("  recent  [--limit <n>] [--path <file>]");
    eprintln!("  compact [--keep <n>] [--path <file>] [--vec-path <file>]");
    eprintln!("\nDefaults:");
    eprintln!("  kind=summary, weight=1.0, limit=3, keep=5000");
}

fn cmd_add(args: &[String]) -> Result<(), &'static str> {
    let mut text: Option<String> = None;
    let mut kind = "summary".to_string();
    let mut weight: f32 = 1.0;
    let mut path = default_path();
    let mut vec_path = default_vec_path();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--text" => {
                i += 1;
                text = args.get(i).cloned();
            }
            "--kind" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    kind = v.clone();
                }
            }
            "--weight" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    weight = v.parse().unwrap_or(1.0);
                }
            }
            "--path" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    path = PathBuf::from(v);
                }
            }
            "--vec-path" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    vec_path = PathBuf::from(v);
                }
            }
            _ => {}
        }
        i += 1;
    }

    let Some(text) = text else {
        eprintln!("Missing --text");
        return Err("missing text");
    };

    ensure_parent_dir(&path).map_err(|_| "mkdir failed")?;
    let record = Record {
        id: now_millis(),
        ts: now_secs(),
        kind,
        weight,
        text,
    };
    append_record(&path, &record).map_err(|_| "write failed")?;
    append_vector(&vec_path, &record).map_err(|_| "write failed")?;
    Ok(())
}

fn cmd_search(args: &[String]) -> Result<(), &'static str> {
    let mut query: Option<String> = None;
    let mut limit: usize = 3;
    let mut path = default_path();
    let mut vec_path = default_vec_path();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--query" => {
                i += 1;
                query = args.get(i).cloned();
            }
            "--limit" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    limit = v.parse().unwrap_or(3);
                }
            }
            "--path" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    path = PathBuf::from(v);
                }
            }
            "--vec-path" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    vec_path = PathBuf::from(v);
                }
            }
            _ => {}
        }
        i += 1;
    }

    let Some(query) = query else {
        eprintln!("Missing --query");
        return Err("missing query");
    };

    let records = load_records(&path).map_err(|_| "read failed")?;
    let vectors = load_vectors(&vec_path).unwrap_or_default();
    let scored = score_records(&query, &records, &vectors, limit);
    for (score, rec) in scored.into_iter().take(limit) {
        println!("{score:.3}\t{}\t{}\t{}\t{}", rec.kind, rec.id, rec.ts, rec.text.replace('\n', " "));
    }
    Ok(())
}

fn cmd_recent(args: &[String]) -> Result<(), &'static str> {
    let mut limit: usize = 20;
    let mut path = default_path();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--limit" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    limit = v.parse().unwrap_or(20);
                }
            }
            "--path" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    path = PathBuf::from(v);
                }
            }
            _ => {}
        }
        i += 1;
    }

    let mut records = load_records(&path).map_err(|_| "read failed")?;
    records.sort_by(|a, b| b.ts.cmp(&a.ts));
    for rec in records.into_iter().take(limit) {
        println!("{}\t{}\t{}\t{}", rec.kind, rec.id, rec.ts, rec.text.replace('\n', " "));
    }
    Ok(())
}

fn cmd_compact(args: &[String]) -> Result<(), &'static str> {
    let mut keep: usize = 5000;
    let mut path = default_path();
    let mut vec_path = default_vec_path();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--keep" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    keep = v.parse().unwrap_or(5000);
                }
            }
            "--path" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    path = PathBuf::from(v);
                }
            }
            "--vec-path" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    vec_path = PathBuf::from(v);
                }
            }
            _ => {}
        }
        i += 1;
    }

    let mut records = load_records(&path).map_err(|_| "read failed")?;
    records.sort_by(|a, b| b.ts.cmp(&a.ts));
    if records.len() > keep {
        records.truncate(keep);
    }
    write_records(&path, &records).map_err(|_| "write failed")?;
    write_vectors(&vec_path, &records).map_err(|_| "write failed")?;
    Ok(())
}

fn default_path() -> PathBuf {
    if let Ok(p) = env::var("MEMSTORE_PATH") {
        return PathBuf::from(p);
    }
    PathBuf::from("memory/memories.log")
}

fn default_vec_path() -> PathBuf {
    if let Ok(p) = env::var("MEMSTORE_VEC_PATH") {
        return PathBuf::from(p);
    }
    PathBuf::from("memory/memories.vec")
}

fn ensure_parent_dir(path: &Path) -> io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    Ok(())
}

fn append_record(path: &Path, rec: &Record) -> io::Result<()> {
    let file = OpenOptionsExt::open_append(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "{}|{}|{}|{}|{}",
        rec.id,
        rec.ts,
        escape(&rec.kind),
        rec.weight,
        escape(&rec.text)
    )?;
    writer.flush()?;
    Ok(())
}

fn append_vector(path: &Path, rec: &Record) -> io::Result<()> {
    let vec = embed_text(&rec.text);
    let file = OpenOptionsExt::open_append(path)?;
    let mut writer = BufWriter::new(file);
    write!(writer, "{}|{}|", rec.id, VECTOR_DIM)?;
    for (i, v) in vec.iter().enumerate() {
        if i > 0 {
            write!(writer, ",")?;
        }
        write!(writer, "{:.6}", v)?;
    }
    writeln!(writer)?;
    writer.flush()?;
    Ok(())
}

fn write_records(path: &Path, records: &[Record]) -> io::Result<()> {
    ensure_parent_dir(path)?;
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for rec in records {
        writeln!(
            writer,
            "{}|{}|{}|{}|{}",
            rec.id,
            rec.ts,
            escape(&rec.kind),
            rec.weight,
            escape(&rec.text)
        )?;
    }
    writer.flush()?;
    Ok(())
}

fn write_vectors(path: &Path, records: &[Record]) -> io::Result<()> {
    ensure_parent_dir(path)?;
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for rec in records {
        let vec = embed_text(&rec.text);
        write!(writer, "{}|{}|", rec.id, VECTOR_DIM)?;
        for (i, v) in vec.iter().enumerate() {
            if i > 0 {
                write!(writer, ",")?;
            }
            write!(writer, "{:.6}", v)?;
        }
        writeln!(writer)?;
    }
    writer.flush()?;
    Ok(())
}

fn load_records(path: &Path) -> io::Result<Vec<Record>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Some(rec) = parse_record_line(&line) {
            records.push(rec);
        }
    }
    Ok(records)
}

fn load_vectors(path: &Path) -> io::Result<HashMap<u128, Vec<f32>>> {
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut map = HashMap::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Some((id, vec)) = parse_vector_line(&line) {
            map.insert(id, vec);
        }
    }
    Ok(map)
}

fn parse_record_line(line: &str) -> Option<Record> {
    let parts: Vec<&str> = line.splitn(5, '|').collect();
    if parts.len() != 5 {
        return None;
    }
    let id = parts[0].parse::<u128>().ok()?;
    let ts = parts[1].parse::<i64>().ok()?;
    let kind = unescape(parts[2]);
    let weight = parts[3].parse::<f32>().ok()?;
    let text = unescape(parts[4]);
    Some(Record {
        id,
        ts,
        kind,
        weight,
        text,
    })
}

fn parse_vector_line(line: &str) -> Option<(u128, Vec<f32>)> {
    let parts: Vec<&str> = line.splitn(3, '|').collect();
    if parts.len() != 3 {
        return None;
    }
    let id = parts[0].parse::<u128>().ok()?;
    let dim = parts[1].parse::<usize>().ok()?;
    let values: Vec<f32> = if parts[2].is_empty() {
        Vec::new()
    } else {
        parts[2]
            .split(',')
            .filter_map(|v| v.parse::<f32>().ok())
            .collect()
    };
    if values.len() != dim {
        return None;
    }
    Some((id, values))
}

fn escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '|' => out.push_str("\\|"),
            _ => out.push(ch),
        }
    }
    out
}

fn unescape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            if let Some(next) = chars.next() {
                match next {
                    'n' => out.push('\n'),
                    '|' => out.push('|'),
                    '\\' => out.push('\\'),
                    _ => {
                        out.push('\\');
                        out.push(next);
                    }
                }
            } else {
                out.push('\\');
            }
        } else {
            out.push(ch);
        }
    }
    out
}

fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut buf = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            buf.push(ch.to_ascii_lowercase());
        } else if !buf.is_empty() {
            tokens.push(buf.clone());
            buf.clear();
        }
    }
    if !buf.is_empty() {
        tokens.push(buf);
    }
    tokens
}

fn score_records(
    query: &str,
    records: &[Record],
    vectors: &HashMap<u128, Vec<f32>>,
    limit: usize,
) -> Vec<(f32, Record)> {
    let query_vec = embed_text(query);
    let now = now_secs();
    let (ids, vecs) = collect_vectors(records, vectors);
    let candidate_ids = hnsw_candidate_ids(&query_vec, &ids, &vecs, records.len(), limit);
    let mut scored: Vec<(f32, Record)> = records
        .iter()
        .filter(|rec| candidate_ids.contains(&rec.id))
        .map(|rec| {
            let vec = vectors
                .get(&rec.id)
                .cloned()
                .unwrap_or_else(|| embed_text(&rec.text));
            let cosine = cosine_sim(&query_vec, &vec);
            let age_days = ((now - rec.ts).max(0) as f32) / 86400.0;
            let recency = 1.0 / (1.0 + age_days);
            let score = cosine * 2.0 + rec.weight * 0.5 + recency;
            (score, rec.clone())
        })
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    scored
}

fn collect_vectors(
    records: &[Record],
    vectors: &HashMap<u128, Vec<f32>>,
) -> (Vec<u128>, Vec<Vec<f32>>) {
    let mut ids = Vec::with_capacity(records.len());
    let mut vecs = Vec::with_capacity(records.len());
    for rec in records {
        let vec = vectors
            .get(&rec.id)
            .cloned()
            .unwrap_or_else(|| embed_text(&rec.text));
        ids.push(rec.id);
        vecs.push(vec);
    }
    (ids, vecs)
}

fn hnsw_candidate_ids(
    query_vec: &[f32],
    ids: &[u128],
    vecs: &[Vec<f32>],
    total: usize,
    limit: usize,
) -> HashSet<u128> {
    let mut set = HashSet::new();
    if vecs.is_empty() {
        return set;
    }
    let k = (limit.saturating_mul(10)).max(10).min(vecs.len());
    if total <= k {
        for id in ids {
            set.insert(*id);
        }
        return set;
    }

    let hnsw: Hnsw<f32, DistCosine> = Hnsw::new(
        HNSW_M,
        vecs.len(),
        HNSW_NB_LAYER,
        HNSW_EF_CONSTRUCTION,
        DistCosine::default(),
    );
    for (i, v) in vecs.iter().enumerate() {
        hnsw.insert((v.as_slice(), i));
    }

    let neighbours: Vec<Neighbour> = hnsw.search(query_vec, k, HNSW_EF_SEARCH.max(k));
    for n in neighbours {
        if let Some(id) = ids.get(n.d_id) {
            set.insert(*id);
        }
    }
    set
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

struct OpenOptionsExt;

impl OpenOptionsExt {
    fn open_append(path: &Path) -> io::Result<File> {
        ensure_parent_dir(path)?;
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
    }
}

fn embed_text(text: &str) -> Vec<f32> {
    let mut vec = vec![0.0f32; VECTOR_DIM];
    let tokens = tokenize(text);
    if tokens.is_empty() {
        return vec;
    }
    for token in tokens.iter() {
        let idx = (fnv1a_hash(token) % VECTOR_DIM as u64) as usize;
        vec[idx] += 1.0;
    }
    normalize(&mut vec);
    vec
}

fn normalize(vec: &mut [f32]) {
    let mut sum = 0.0f32;
    for v in vec.iter() {
        sum += v * v;
    }
    let norm = sum.sqrt();
    if norm > 0.0 {
        for v in vec.iter_mut() {
            *v /= norm;
        }
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut a_norm = 0.0f32;
    let mut b_norm = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        a_norm += a[i] * a[i];
        b_norm += b[i] * b[i];
    }
    if a_norm == 0.0 || b_norm == 0.0 {
        0.0
    } else {
        dot / (a_norm.sqrt() * b_norm.sqrt())
    }
}

fn fnv1a_hash(input: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in input.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
