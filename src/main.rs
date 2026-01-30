use std::cmp::Ordering;
use std::collections::HashSet;
use std::env;
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use hnsw_rs::anndists::dist::distances::DistCosine;
use hnsw_rs::prelude::{Hnsw, Neighbour};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Record {
    id: u128,
    ts: i64,
    kind: String,
    weight: f32,
    text: String,
    vector: Vec<f32>,
}

const VECTOR_DIM: usize = 256;
const STORE_VERSION: u32 = 1;
const HNSW_M: usize = 16;
const HNSW_EF_CONSTRUCTION: usize = 200;
const HNSW_NB_LAYER: usize = 16;
const HNSW_EF_SEARCH: usize = 50;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Store {
    version: u32,
    vector_dim: usize,
    records: Vec<Record>,
}

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
    eprintln!("  add     --text <text> [--kind <kind>] [--weight <w>] [--path <file>]");
    eprintln!("  search  --query <text> [--limit <n>] [--path <file>]");
    eprintln!("  recent  [--limit <n>] [--path <file>]");
    eprintln!("  compact [--keep <n>] [--path <file>]");
    eprintln!("\nDefaults:");
    eprintln!("  kind=summary, weight=1.0, limit=3, keep=5000");
}

fn cmd_add(args: &[String]) -> Result<(), &'static str> {
    let mut text: Option<String> = None;
    let mut kind = "summary".to_string();
    let mut weight: f32 = 1.0;
    let mut path = default_path();

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
        vector: embed_text(&text),
        text,
    };
    let mut store = load_store(&path).map_err(|_| "read failed")?;
    store.records.push(record);
    save_store(&path, &store).map_err(|_| "write failed")?;
    Ok(())
}

fn cmd_search(args: &[String]) -> Result<(), &'static str> {
    let mut query: Option<String> = None;
    let mut limit: usize = 3;
    let mut path = default_path();

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
            _ => {}
        }
        i += 1;
    }

    let Some(query) = query else {
        eprintln!("Missing --query");
        return Err("missing query");
    };

    let store = load_store(&path).map_err(|_| "read failed")?;
    let scored = score_records(&query, &store.records, limit);
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

    let store = load_store(&path).map_err(|_| "read failed")?;
    let mut records = store.records;
    records.sort_by(|a, b| b.ts.cmp(&a.ts));
    for rec in records.into_iter().take(limit) {
        println!("{}\t{}\t{}\t{}", rec.kind, rec.id, rec.ts, rec.text.replace('\n', " "));
    }
    Ok(())
}

fn cmd_compact(args: &[String]) -> Result<(), &'static str> {
    let mut keep: usize = 5000;
    let mut path = default_path();

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
            _ => {}
        }
        i += 1;
    }

    let mut store = load_store(&path).map_err(|_| "read failed")?;
    store.records.sort_by(|a, b| b.ts.cmp(&a.ts));
    if store.records.len() > keep {
        store.records.truncate(keep);
    }
    save_store(&path, &store).map_err(|_| "write failed")?;
    Ok(())
}

fn default_path() -> PathBuf {
    if let Ok(p) = env::var("MEMSTORE_PATH") {
        return PathBuf::from(p);
    }
    PathBuf::from("memory/memories.hnsw")
}

fn ensure_parent_dir(path: &Path) -> io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    Ok(())
}

fn save_store(path: &Path, store: &Store) -> io::Result<()> {
    ensure_parent_dir(path)?;
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let data = bincode::serialize(store).map_err(|_| io::ErrorKind::InvalidData)?;
    writer.write_all(&data)?;
    writer.flush()?;
    Ok(())
}

fn load_store(path: &Path) -> io::Result<Store> {
    if !path.exists() {
        return Ok(Store {
            version: STORE_VERSION,
            vector_dim: VECTOR_DIM,
            records: Vec::new(),
        });
    }
    let file = File::open(path)?;
    let mut data = Vec::new();
    let mut reader = file;
    reader.read_to_end(&mut data)?;
    let store: Store = bincode::deserialize(&data).map_err(|_| io::ErrorKind::InvalidData)?;
    if store.vector_dim != VECTOR_DIM || store.version != STORE_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "incompatible store format",
        ));
    }
    Ok(store)
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

fn score_records(query: &str, records: &[Record], limit: usize) -> Vec<(f32, Record)> {
    let query_vec = embed_text(query);
    let now = now_secs();
    let (indices, vecs) = collect_vectors(records);
    let candidate_indices =
        hnsw_candidate_indices(&query_vec, &indices, &vecs, records.len(), limit);
    let mut scored: Vec<(f32, Record)> = records
        .iter()
        .enumerate()
        .filter(|(idx, _)| candidate_indices.contains(idx))
        .map(|(_, rec)| {
            let cosine = cosine_sim(&query_vec, &rec.vector);
            let age_days = ((now - rec.ts).max(0) as f32) / 86400.0;
            let recency = 1.0 / (1.0 + age_days);
            let score = cosine * 2.0 + rec.weight * 0.5 + recency;
            (score, rec.clone())
        })
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    scored
}

fn collect_vectors(records: &[Record]) -> (Vec<usize>, Vec<Vec<f32>>) {
    let mut indices = Vec::with_capacity(records.len());
    let mut vecs = Vec::with_capacity(records.len());
    for (i, rec) in records.iter().enumerate() {
        indices.push(i);
        vecs.push(rec.vector.clone());
    }
    (indices, vecs)
}

fn hnsw_candidate_indices(
    query_vec: &[f32],
    indices: &[usize],
    vecs: &[Vec<f32>],
    total: usize,
    limit: usize,
) -> HashSet<usize> {
    let mut set = HashSet::new();
    if vecs.is_empty() {
        return set;
    }
    let k = (limit.saturating_mul(10)).max(10).min(vecs.len());
    if total <= k {
        for idx in indices {
            set.insert(*idx);
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
        if let Some(idx) = indices.get(n.d_id) {
            set.insert(*idx);
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
