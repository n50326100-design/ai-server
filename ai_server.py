import os, time, asyncio
from collections import defaultdict
import aiohttp
from aiohttp import web

GEMINI_KEY = os.environ["GEMINI_API_KEY"]
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={GEMINI_KEY}"

TEXT_LIMIT = 30
IMG_LIMIT = 10
MAX_IMAGE_CHARS = 3_000_000
RETRIES = 2

CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
}

usage = defaultdict(lambda: {"day": "", "text": 0, "img": 0})

SYSTEM = (
    "Sen o'zbekistonlik maktab o'quvchilari uchun fan o'qituvchisisan. "
    "Savolga aniq, qadamma-qadam va sodda tilda javob ber. Misollar bilan tushuntir. "
    "Savol maktab fanlariga aloqador bo'lmasa, muloyimlik bilan fanga qaytar. "
    "Javobni foydalanuvchi tilida (uz, ru yoki en) ber."
)


def client_ip(request):
    fwd = request.headers.get("X-Forwarded-For")
    return fwd.split(",")[0].strip() if fwd else request.remote


def allowed(ip, kind):
    today = time.strftime("%Y-%m-%d")
    u = usage[ip]
    if u["day"] != today:
        u.update(day=today, text=0, img=0)
    limit = IMG_LIMIT if kind == "img" else TEXT_LIMIT
    if u[kind] >= limit:
        return False
    u[kind] += 1
    return True


def json_error(msg, status):
    return web.json_response({"error": msg}, status=status, headers=CORS)


async def call_gemini(body):
    """Gemini'ga so'rov. 503 (band) bo'lsa bir necha marta kutib qayta urinadi."""
    last_status, last_res = None, None
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as s:
        for attempt in range(RETRIES + 1):
            async with s.post(URL, json=body) as r:
                res = await r.json(content_type=None)
                if r.status == 200:
                    return res, 200
                last_status, last_res = r.status, res
                print("GEMINI ERROR", r.status, res, flush=True)
                if r.status not in (429, 500, 503):
                    break
            await asyncio.sleep(3 * (attempt + 1))
    return last_res, last_status


async def ask(request):
    if request.method == "OPTIONS":
        return web.Response(headers=CORS)

    try:
        data = await request.json()
    except Exception:
        return json_error("bad_request", 400)

    lang = str(data.get("lang", "uz"))[:5]
    subject = str(data.get("subject", ""))[:50]
    question = str(data.get("question", ""))[:1000].strip()
    image = data.get("image")
    ip = client_ip(request)

    parts = []
    if image:
        if not allowed(ip, "img"):
            return json_error("limit", 429)
        if len(image) > MAX_IMAGE_CHARS:
            return json_error("too_big", 413)
        parts.append({"inline_data": {
            "mime_type": "image/jpeg",
            "data": image.split(",", 1)[-1],
        }})
        parts.append({"text": question or
                      "Rasmda nima borligini maktab o'quvchisi tushunadigan qilib tushuntiring."})
    else:
        if not question:
            return json_error("empty", 400)
        if not allowed(ip, "text"):
            return json_error("limit", 429)
        parts.append({"text": f"Fan: {subject}\nTil: {lang}\nSavol: {question}"})

    body = {
        "system_instruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"maxOutputTokens": 800},
    }

    try:
        res, status = await call_gemini(body)
    except Exception as e:
        print("REQUEST ERROR", repr(e), flush=True)
        return json_error("ai_error", 500)

    if status != 200:
        return json_error("ai_busy" if status in (429, 503) else "ai_error", 503 if status in (429, 503) else 500)

    try:
        answer = res["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        print("BAD RESPONSE", res, flush=True)
        answer = "Javob olib bo'lmadi. Savolni boshqacha yozib ko'ring."
    return web.json_response({"answer": answer}, headers=CORS)


async def health(request):
    return web.Response(text="ok", headers=CORS)


app = web.Application(client_max_size=5 * 1024 * 1024)
app.router.add_route("*", "/ask", ask)
app.router.add_get("/", health)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    web.run_app(app, host="0.0.0.0", port=port)