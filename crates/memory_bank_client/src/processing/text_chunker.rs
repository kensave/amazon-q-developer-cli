/// Chunk text into smaller pieces with overlap
///
/// # Arguments
///
/// * `text` - The text to chunk
/// * `chunk_size` - The size of each chunk in words
/// * `overlap` - The number of words to overlap between chunks
///
/// # Returns
///
/// A vector of string chunks
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let words: Vec<&str> = text.split_whitespace().collect();

    if words.is_empty() {
        return chunks;
    }

    let mut i = 0;
    while i < words.len() {
        let end = (i + chunk_size).min(words.len());
        let chunk = words[i..end].join(" ");
        chunks.push(chunk);

        // Move forward by chunk_size - overlap
        i += chunk_size - overlap;
        if i >= words.len() || i == 0 {
            break;
        }
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_text_empty() {
        let chunks = chunk_text("", 100, 20);
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn test_chunk_text_small() {
        let text = "This is a small text";
        let chunks = chunk_text(text, 10, 2);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn test_chunk_text_large() {
        let words: Vec<String> = (0..200).map(|i| format!("word{}", i)).collect();
        let text = words.join(" ");

        let chunks = chunk_text(&text, 50, 10);

        // With 200 words, chunk size 50, and overlap 10, we should have 5 chunks
        // (0-49, 40-89, 80-129, 120-169, 160-199)
        assert_eq!(chunks.len(), 5);

        // Check first and last words of first chunk
        assert!(chunks[0].starts_with("word0"));
        assert!(chunks[0].ends_with("word49"));

        // Check first and last words of last chunk
        assert!(chunks[4].starts_with("word160"));
        assert!(chunks[4].ends_with("word199"));
    }
}
