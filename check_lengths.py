from sentence_transformers import SentenceTransformer
from utils import iter_entries, entry_text, MAX_SEQ_LENGTH, EMBEDDING_MODEL_NAME

def check_corpus_lengths():
    # טעינת המודל כדי להשתמש ב-Tokenizer שלו (הדרך המדויקת ביותר)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    tokenizer = model.tokenizer
    
    long_pages = []
    
    print(f"Checking corpus against MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}...")
    
    for record in iter_entries():
        text = entry_text(record)
        # חישוב כמות הטוקנים
        tokens = tokenizer.encode(text)
        token_count = len(tokens)
        
        if token_count > MAX_SEQ_LENGTH:
            long_pages.append((record['page_id'], token_count))
    
    # סיכום תוצאות
    if long_pages:
        print(f"Found {len(long_pages)} pages exceeding limit:")
        for pid, count in sorted(long_pages, key=lambda x: x[1], reverse=True)[:10]:
            print(f"Page ID {pid}: {count} tokens")
    else:
        print("Success! No pages exceed the MAX_SEQ_LENGTH.")

if __name__ == "__main__":
    check_corpus_lengths()