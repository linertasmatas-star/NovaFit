# Sukuria 3D drabužių miniatiūras: assets/thumbs3d/<lytis>-<tipas>.png (balti, be fono; spalva uždedama programėlėje).
# Reikia: vietinio serverio (python3 -m http.server 8765 repozitorijos šaknyje) ir Playwright su Chrome.
import asyncio, base64, os, sys
from playwright.async_api import async_playwright
TYPES = sys.argv[1:] or ["hoodie", "sweater", "jeans"]
async def main():
    os.makedirs("assets/thumbs3d", exist_ok=True)
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        pg = await (await b.new_context()).new_page()
        pg.on("pageerror", lambda e: print("klaida:", e))
        await pg.goto("http://localhost:8765/tools/thumbs3d.html")
        await pg.wait_for_function("window.ready === true")
        for t in TYPES:
            for g in ("female", "male"):
                url = await pg.evaluate(f"drawThumb('{t}', '{g}')")
                open(f"assets/thumbs3d/{g}-{t}.png", "wb").write(base64.b64decode(url.split(",")[1]))
                print("ok", g, t)
        await b.close()
asyncio.run(main())
