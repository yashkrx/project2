import tempfile
import zipfile
import tarfile
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import base64
import io
from dotenv import load_dotenv
import google.genai as genai

load_dotenv()

app = FastAPI()

# Environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)


def get_content_type_for_image(filename):
    ext = filename.lower().split('.')[-1]
    if ext in ["jpg", "jpeg"]:
        return "image/jpeg"
    elif ext == "png":
        return "image/png"
    elif ext == "gif":
        return "image/gif"
    else:
        return "application/octet-stream"


class FileData:
    def __init__(self, name, content, content_type, is_image=False, is_text=False):
        self.name = name
        self.content = content
        self.content_type = content_type
        self.is_image = is_image
        self.is_text = is_text


@app.post("/api/")
async def analyze_data(request: Request):
    form = await request.form()

    if "questions.txt" not in form:
        return JSONResponse(status_code=400, content={"message": "questions.txt is required"})

    questions_content = ""
    processed_files = []
    files_info_response = {}

    # Use a unique temp dir per request (TemporaryDirectory already does this)
    with tempfile.TemporaryDirectory() as temp_dir:
        for name, file_or_field in form.items():
            if not hasattr(file_or_field, 'filename'):
                files_info_response[name] = str(file_or_field)
                continue

            file = file_or_field
            content = await file.read()

            if name == "questions.txt":
                try:
                    questions_content = content.decode('utf-8')
                except Exception:
                    return JSONResponse(status_code=400, content={"message": "questions.txt must be UTF-8 text"})
                files_info_response[name] = {"filename": file.filename, "content_preview": questions_content[:200]}
                continue

            is_zip = file.filename.lower().endswith('.zip')
            is_tar = file.filename.lower().endswith(('.tar', '.tar.gz', '.tgz'))

            if is_zip or is_tar:
                files_info_response[name] = {"filename": file.filename, "extracted_files": []}
                archive_path = io.BytesIO(content)
                try:
                    # extract into a subdir so archives don't overwrite each other
                    extract_subdir = os.path.join(temp_dir, f"extracted_{name}")
                    os.makedirs(extract_subdir, exist_ok=True)

                    if is_zip:
                        with zipfile.ZipFile(archive_path) as zf:
                            zf.extractall(extract_subdir)
                    elif is_tar:
                        with tarfile.open(fileobj=archive_path) as tf:
                            tf.extractall(extract_subdir)

                    for root, _, filenames in os.walk(extract_subdir):
                        for extracted_filename in filenames:
                            extracted_filepath = os.path.join(root, extracted_filename)
                            with open(extracted_filepath, 'rb') as f:
                                extracted_content = f.read()

                            is_image = extracted_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))
                            content_type = get_content_type_for_image(extracted_filename) if is_image else 'text/plain'

                            processed_files.append(FileData(
                                name=extracted_filename,
                                content=extracted_content,
                                content_type=content_type,
                                is_image=is_image,
                                is_text=not is_image
                            ))
                            files_info_response[name]["extracted_files"].append({"filename": extracted_filename})

                except (zipfile.BadZipFile, tarfile.ReadError) as e:
                    files_info_response[name]["error"] = f"Could not decompress file: {e}"
            else:
                is_image = bool(file.content_type and file.content_type.startswith('image/'))
                processed_files.append(FileData(
                    name=file.filename,
                    content=content,
                    content_type=file.content_type,
                    is_image=is_image,
                    is_text=not is_image
                ))
                files_info_response[name] = {"filename": file.filename}

    # Build a single prompt string (safer and simpler than multipart API usage)
    prompt_parts = []
    if questions_content:
        prompt_parts.append("Here is the question I need you to answer:\n---\n")
        prompt_parts.append(questions_content)
        prompt_parts.append("\n---\nTo answer the question, you have access to the following files:\n")

        for p_file in processed_files:
            if p_file.is_image:
                # include a short descriptor + base64 length, avoid embedding huge images fully
                b64 = base64.b64encode(p_file.content).decode('utf-8')
                # only include data URI if it's small; otherwise include length so the model knows it's an image
                if len(b64) < 20000:  # ~15KB binary threshold
                    data_uri = f"data:{p_file.content_type};base64,{b64}"
                    prompt_parts.append(f"- An image file named `{p_file.name}` with data URI: {data_uri}\n")
                else:
                    prompt_parts.append(f"- An image file named `{p_file.name}` (content_type={p_file.content_type}, base64_length={len(b64)})\n")
            elif p_file.is_text:
                try:
                    decoded_content = p_file.content.decode('utf-8')
                    preview = "\n".join(decoded_content.splitlines()[:40])
                    prompt_parts.append(f"- A text file named `{p_file.name}`. Preview:\n---\n{preview}\n---\n")
                except Exception:
                    prompt_parts.append(f"- A file named `{p_file.name}` which appears to be binary.\n")

    full_prompt = "\n".join(prompt_parts)

    generated_code = "# No valid question provided or an error occurred."
    if full_prompt.strip():
        system_prompt = (
               "You are a world-class data analyst AI. Your purpose is to write robust, production-quality Python code to solve a user's question based on the data files they provide. "
            "You must follow these instructions meticulously:\n"
            "1.  **Analyze the Request:** Carefully read the user's question and examine the previews of all provided files (text, CSV, images, etc.) to understand the context and requirements fully."
            "2.  **Think Step-by-Step:** Before writing code, formulate a clear plan. Consider data loading, necessary cleaning (handling missing values, correcting data types, ensuring case consistency), analysis steps, and the final output format."
            "3.  **Write High-Quality Python Codeః"
            "    - The code must be pure Python and executable. Assume standard libraries like `pandas`, `matplotlib`, `numpy`, and `base64` are available."
            "    - Refer to files by their exact filenames as provided in the prompt (e.g., `sample-sales.csv`). Do not invent or assume file paths."
            "    - **Crucial:** Perform data cleaning and preprocessing. Do not make assumptions about data quality. Check for and handle inconsistencies."
            "    - Your code must print the final answer(s) to standard output. The output format must precisely match what the user requested."
            "    - If the question requires creating a plot or image, you MUST save it to a file (e.g., `plot.png`) and then print its base64 data URI to standard output (e.g., `print(f'data:image/png;base64,{base64_string}')`)."
            "4.  **Final Output:** Your response MUST contain ONLY the raw Python code. Do not include any explanations, comments, or markdown formatting like ```python ... ```. Just the code itself."     )

        # Combine into a single content message for Gemini
        content_message = system_prompt + "\n\n" + full_prompt

        try:
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=[
                    {"role": "user", "parts": [content_message]}
                ],
                max_output_tokens=32768,
            )

            # Robust extraction of text from response
            generated_code = None
            try:
                # normal path
                generated_code = response.candidates[0].content.parts[0].text
            except Exception:
                # fallback to stringifying the response if structure differs
                try:
                    generated_code = str(response)
                except Exception as e:
                    generated_code = f"# Failed to parse Gemini response: {e}"

        except Exception as e:
            generated_code = f"# An error occurred while contacting the Gemini API: {e}"

    return JSONResponse(content={"files_processed": files_info_response, "generated_code": generated_code})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
