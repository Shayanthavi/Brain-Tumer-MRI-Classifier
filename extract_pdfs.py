import pypdf

def extract_pdf_text(pdf_path, txt_path):
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for i, page in enumerate(reader.pages):
            text += f"--- Page {i+1} ---\n"
            text += page.extract_text() + "\n"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Extracted {len(reader.pages)} pages from {pdf_path} to {txt_path}")
    except Exception as e:
        print(f"Error extracting {pdf_path}: {e}")

extract_pdf_text("Proposal.pdf", "Proposal.txt")
extract_pdf_text("Conference_Paper_Review_Group_Number_03.pdf", "Conference_Paper.txt")
