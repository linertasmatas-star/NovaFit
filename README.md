# NovaFit

Aprangos derinių kūrimo programėlė – Junior Achievement verslumo konkurso projektas.

## Kaip paleisti
Atsisiųskite visą repozitoriją (Code → Download ZIP), išskleiskite ir du kartus spustelėkite `index.html`. Aplankas `assets` turi būti šalia `index.html`. Nieko diegti nereikia.

## Gyva nuoroda
https://linertasmatas-star.github.io/NovaFit/

## Kas jau veikia (demo v8)
- Pradžios žingsniai (rodomi pirmą kartą, vėliau – paspaudus logotipą): Sveiki → Paskyra → Matmenys → Avataras. Paskyra – demonstracinė imitacija: paskyros nekuriamos, duomenys niekur nesiunčiami. Papildomi matmenys (rankų ilgis, juosmuo, klubai) išsaugomi, bet avataro kol kas nekeičia
- Naujas pagrindinis ekranas pagal dizaino sistemą: šviesi scena su avataru, apatinis meniu (Derinys / Spinta / Mano deriniai + „Pasiūlyk derinį“), matomo drabužio spalvos peržiūra
- Avataro baziniai apatiniai – juodi
- Laisvo kirpimo drabužiai: laisvi (oversize) marškinėliai ir plačios kelnės – visoms 10 figūrų, kataloge ir kaip tipai įkeliant savo drabužį
- Remiamos rekomendacijos: stilistas šalia derinio parodo vieną prekės ženklo drabužį su žyma „Remiama“ – galima pasimatuoti ant avataro ir „Žiūrėti parduotuvėje“. Prekės ženklas, kainos ir parduotuvė kol kas išgalvoti (demonstracinė imitacija)
- Stilistas: pasirinkus progą (mokykla, sportas, šventė) derinys užsideda ant avataro – pirmiausia iš savo spintos, ko trūksta – iš katalogo; trumpas paaiškinimas ir „Kitas variantas“. Veikia pagal progos ir spalvų derinimo taisykles – tai ne dirbtinis intelektas (demonstracinė imitacija)
- Titulinis ekranas; katalogo paveikslėliai iš tų pačių sluoksnių, kurie dedami ant avataro
- Drabužių katalogas (15 pavyzdinių drabužių) su filtru pagal kategoriją
- Tikroviškas avataras iš MakeHuman figūrų: lytis (moteris/vyras), 5 svorio lygiai, mastelis pagal ūgį
- Drabužiai „pasiūti“ ant 3D kūno kiekvienai figūrai, spalva uždedama naršyklėje
- Derinių išsaugojimas su pavadinimu, sąrašas su mažais avatarais, atidarymas, ištrynimas
- Programėlė atsimena lytį, ūgį, svorį ir dabartinį derinį (naršyklės atmintis – tik šiame kompiuteryje)
- Drabužių uždėjimas ant avataro vienu paspaudimu, nuėmimas, „Nurengti viską“
- „Mano spinta“: savo drabužio nuotraukos įkėlimas, tipą pasirenka vartotojas, spalva paimama iš nuotraukos automatiškai (galima pataisyti). Ant avataro uždedamas mūsų paruoštas to tipo ir spalvos sluoksnis, ne pati nuotrauka (demonstracinė imitacija)

## Kas bus toliau
- Fono šalinimas iš nuotraukų

## Kur keisti drabužius
`index.html` faile, skiltyje „1. DUOMENYS“ – kiekviena eilutė `CATALOG` sąraše yra vienas drabužis.

## Avataro paveikslėliai
`assets/avatar/<figūra>/` – `body.png` ir po vieną pilką sluoksnį kiekvienam drabužio tipui. Sugeneruota skriptu `tools/mh_wardrobe.py` iš MakeHuman (CC0) duomenų. Norint pridėti naują drabužio tipą, jį reikia aprašyti skripte ir sugeneruoti iš naujo.

Kaip sugeneruoti iš naujo:
1. Atsisiųsti MakeHuman duomenis: `github.com/makehumancommunity/mpfb2`, aplankas `src/mpfb/data` (reikia `3dobjs` ir `targets/macrodetails`).
2. `python3 tools/mh_wardrobe.py <kelias iki src/mpfb/data> assets/avatar`
3. `python3 tools/smooth_shoes.py` (batai be pirštų).

Aptempti drabužiai „siuvami“ ant 3D kūno (`GARMENTS` sąrašas skripte). Laisvo kirpimo drabužiai (oversize marškinėliai, plačios kelnės) aprašomi kaip kontūras ir iškilumas (`SOFT` sąrašas, skiltis 5b).
