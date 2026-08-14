from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import tempfile
import os

from .service import verify_image

app = FastAPI(title="LaQris QRIS Verification API")


@app.get("/health")
async def health():
	return {"status": "ok"}


@app.post("/verify")
async def verify(file: UploadFile = File(...)):
	"""
	Terima upload gambar, simpan sementara, jalankan verifier, kembalikan hasil.
	Saat ini fungsi verifier menjalankan skrip `file_test.py` sebagai subprocess
	(lazy integration). Nanti kita bisa ganti menjadi in-process loader singleton.
	"""
	suffix = os.path.splitext(file.filename)[1]
	with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
		tmp_path = tmp.name
		shutil.copyfileobj(file.file, tmp)

	try:
		result = verify_image(tmp_path)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))
	finally:
		try:
			os.remove(tmp_path)
		except Exception:
			pass

	return JSONResponse(content=result)
