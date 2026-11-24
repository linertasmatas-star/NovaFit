# NovaFit

Aprangos derinių kūrimo programėlė – Junior Achievement verslumo konkurso projektas.

## Kaip paleisti
Atsisiųskite visą repozitoriją (Code → Download ZIP), išskleiskite ir du kartus spustelėkite `index.html`. Aplankas `assets` turi būti šalia `index.html`. Nieko diegti nereikia.

## Gyva nuoroda
https://linertasmatas-star.github.io/NovaFit/

## Kas jau veikia (demo v21)
- Skirtukas „Parduotuvės“ (demonstracinė imitacija): 5 prekės iš 3 išgalvotų parduotuvių su nuotraukomis (`assets/shop/`), filtras pagal parduotuvę, „Pasimatuoti“ uždeda prekę ant avataro (džemperis, megztinis, džinsai, kostiuminės kelnės – 3D modeliai), bendras krepšelis su pristatymu kiekvienai parduotuvei, „Užsakyti vienu kartu“ – niekur nesiunčiama. Prekės – `CATALOG` su `shop:` (id 201–205)
- Ištrynus drabužį ar derinį 5 sekundes rodoma „Ištrinta · Atšaukti“ – galima sugrąžinti (drabužis grįžta ir ant avataro, ir į išsaugotus derinius)
- „Mano spinta“: didesnės drabužių nuotraukos (telefone 3 stulpeliai), kortelė „Įkelti drabužį“ su +; be pavadinimo drabužis vadinamas tipu (pvz. „Džinsai“), po juo – kategorija
- Naujas drabužio įkėlimas: „Įkelti drabužį“ atidaro lapą „Fotografuoti“ (telefone – iškart kamera) / „Iš galerijos“; formoje – didelė nuotrauka su „Kita nuotrauka“, tipas mygtukais pagal grupes (Viršus, Apačia…), spalva iš nuotraukos (paspaudus apskritimą – pasirinkti pačiam), raštas, pavadinimas. Meniu – nauja spintos ikona
- Džemperis su kapišonu – pirmas Blenderyje sukurtas 3D drabužis (`assets/avatar3d/hoodie.glb`): kapišonas, kengūros kišenė, raišteliai, juostos, raukšlės; keičiasi kartu su figūra (tos pačios 13 formų). 2D rodomas megztinio sluoksniu. Kaip sukurti iš naujo: `tools/blender/` (0 – sričių duomenys, 1 – pagrindas, 2 – forma, 3 – detalės; leidžiama Blenderyje per MCP)
- Savo drabužio raštas ant 2D avataro: iš nuotraukos automatiškai išrenkamas audinio kvadratas (be fono), juo užpildomas drabužio sluoksnis (šešėliai išlieka). Kirpimas – pagal pasirinktą tipą (imitacija). Galima išjungti varnele formoje
- 3D avataras (jungiklis 2D / 3D ant scenos): figūra keičiasi pagal lytį, ūgį, svorį, juosmenį, klubus ir rankų ilgį; drabužiai keičiasi kartu su figūra; sukamas pirštu. Kraunamas tik įjungus (~3 MB). Kaip sukurti iš naujo: `python3 tools/mh_export_glb.py <mpfb2 src/mpfb/data> assets/avatar3d` (reikia 3dobjs, targets/macrodetails, targets/torso, targets/arms). Bandymui – `tools/test3d.html` (per vietinį serverį)
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
