//! # Storage Abstraction
//!
//! Provides a unified `Storage` enum that backs index data from different sources:
//! - Memory-mapped files (`Mmap`)
//! - Owned byte buffers (`Vec<u8>`)
//! - Shared byte buffers (`Arc<[u8]>`)
//!
//! All variants deref to `&[u8]`, so callers index into storage uniformly.

use memmap2::Mmap;
use std::ops::Deref;
use std::sync::Arc;

/// Backing store for index data.
///
/// `Storage` replaces the bare `Mmap` field so that indexes can be loaded
/// from files *or* from in-memory byte buffers without duplicating search logic.
pub enum Storage {
    /// Memory-mapped file (zero-copy, OS-managed paging).
    Mmap(Mmap),
    /// Owned byte buffer (e.g. from `to_bytes()` round-trip or network).
    Owned(Vec<u8>),
    /// Reference-counted shared buffer (cheap clone, multi-reader).
    Shared(Arc<[u8]>),
}

impl Deref for Storage {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        match self {
            Storage::Mmap(m) => m,
            Storage::Owned(v) => v,
            Storage::Shared(a) => a,
        }
    }
}

impl AsRef<[u8]> for Storage {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_owned_deref() {
        let data = vec![1u8, 2, 3, 4];
        let s = Storage::Owned(data.clone());
        assert_eq!(&*s, &data[..]);
    }

    #[test]
    fn test_shared_deref() {
        let data: Arc<[u8]> = Arc::from(vec![10u8, 20, 30]);
        let s = Storage::Shared(data.clone());
        assert_eq!(&*s, &*data);
    }
}
