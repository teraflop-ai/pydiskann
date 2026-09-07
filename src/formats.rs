use crate::DiskAnnError;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Read an fvecs file (float vectors).
pub fn read_fvecs<P: AsRef<Path>>(path: P) -> Result<Vec<Vec<f32>>, DiskAnnError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();

    loop {
        let mut dim_buf = [0u8; 4];
        match reader.read_exact(&mut dim_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let dim = u32::from_le_bytes(dim_buf) as usize;

        let mut data = vec![0u8; dim * 4];
        reader.read_exact(&mut data)?;
        let floats: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        vectors.push(floats);
    }

    Ok(vectors)
}

/// Write vectors to an fvecs file.
pub fn write_fvecs<P: AsRef<Path>>(path: P, vectors: &[Vec<f32>]) -> Result<(), DiskAnnError> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    for v in vectors {
        writer.write_all(&(v.len() as u32).to_le_bytes())?;
        for &val in v {
            writer.write_all(&val.to_le_bytes())?;
        }
    }

    writer.flush()?;
    Ok(())
}

/// Read an ivecs file (integer vectors).
pub fn read_ivecs<P: AsRef<Path>>(path: P) -> Result<Vec<Vec<i32>>, DiskAnnError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();

    loop {
        let mut dim_buf = [0u8; 4];
        match reader.read_exact(&mut dim_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let dim = u32::from_le_bytes(dim_buf) as usize;

        let mut data = vec![0u8; dim * 4];
        reader.read_exact(&mut data)?;
        let ints: Vec<i32> = data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        vectors.push(ints);
    }

    Ok(vectors)
}

/// Write vectors to an ivecs file.
pub fn write_ivecs<P: AsRef<Path>>(path: P, vectors: &[Vec<i32>]) -> Result<(), DiskAnnError> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    for v in vectors {
        writer.write_all(&(v.len() as u32).to_le_bytes())?;
        for &val in v {
            writer.write_all(&val.to_le_bytes())?;
        }
    }

    writer.flush()?;
    Ok(())
}

/// Read a bvecs file (byte vectors).
pub fn read_bvecs<P: AsRef<Path>>(path: P) -> Result<Vec<Vec<u8>>, DiskAnnError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();

    loop {
        let mut dim_buf = [0u8; 4];
        match reader.read_exact(&mut dim_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let dim = u32::from_le_bytes(dim_buf) as usize;

        let mut data = vec![0u8; dim];
        reader.read_exact(&mut data)?;
        vectors.push(data);
    }

    Ok(vectors)
}

/// Read a bvecs file and convert to f32 vectors (dividing by 255.0).
pub fn read_bvecs_as_f32<P: AsRef<Path>>(path: P) -> Result<Vec<Vec<f32>>, DiskAnnError> {
    let bvecs = read_bvecs(path)?;
    Ok(bvecs
        .into_iter()
        .map(|v| v.into_iter().map(|b| b as f32 / 255.0).collect())
        .collect())
}

/// Write byte vectors to a bvecs file.
pub fn write_bvecs<P: AsRef<Path>>(path: P, vectors: &[Vec<u8>]) -> Result<(), DiskAnnError> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    for v in vectors {
        writer.write_all(&(v.len() as u32).to_le_bytes())?;
        writer.write_all(v)?;
    }

    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fvecs_round_trip() {
        let path = "test_fvecs_rt.fvecs";
        let vectors = vec![
            vec![1.0f32, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        write_fvecs(path, &vectors).unwrap();
        let loaded = read_fvecs(path).unwrap();

        assert_eq!(vectors, loaded);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_ivecs_round_trip() {
        let path = "test_ivecs_rt.ivecs";
        let vectors = vec![vec![10i32, 20, 30], vec![40, 50, 60]];

        write_ivecs(path, &vectors).unwrap();
        let loaded = read_ivecs(path).unwrap();

        assert_eq!(vectors, loaded);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_bvecs_round_trip() {
        let path = "test_bvecs_rt.bvecs";
        let vectors = vec![vec![0u8, 128, 255], vec![1, 2, 3]];

        write_bvecs(path, &vectors).unwrap();
        let loaded = read_bvecs(path).unwrap();

        assert_eq!(vectors, loaded);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_bvecs_as_f32() {
        let path = "test_bvecs_f32.bvecs";
        let vectors = vec![vec![0u8, 255]];

        write_bvecs(path, &vectors).unwrap();
        let loaded = read_bvecs_as_f32(path).unwrap();

        assert_eq!(loaded.len(), 1);
        assert!((loaded[0][0] - 0.0).abs() < 1e-6);
        assert!((loaded[0][1] - 1.0).abs() < 1e-6);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_fvecs_empty() {
        let path = "test_fvecs_empty.fvecs";
        let vectors: Vec<Vec<f32>> = vec![];

        write_fvecs(path, &vectors).unwrap();
        let loaded = read_fvecs(path).unwrap();

        assert!(loaded.is_empty());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_fvecs_varying_would_fail() {
        // fvecs format allows varying dims per vector (each has its own dim header)
        let path = "test_fvecs_vary.fvecs";
        let vectors = vec![vec![1.0f32, 2.0], vec![3.0, 4.0, 5.0]];

        write_fvecs(path, &vectors).unwrap();
        let loaded = read_fvecs(path).unwrap();

        assert_eq!(loaded[0].len(), 2);
        assert_eq!(loaded[1].len(), 3);
        std::fs::remove_file(path).ok();
    }
}
