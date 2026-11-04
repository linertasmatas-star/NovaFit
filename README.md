# NovaFit

Aprangos derinių kūrimo programėlė – Junior Achievement verslumo konkurso projektas.

## Kaip paleisti
Atsisiųskite visą repozitoriją (Code → Download ZIP), išskleiskite ir du kartus spustelėkite `index.html`. Aplankas `assets` turi būti šalia `index.html`. Nieko diegti nereikia.

## Kas jau veikia (demo v2)
- Drabužių katalogas (15 pavyzdinių drabužių) su filtru pagal kategoriją
- Tikroviškas avataras iš MakeHuman figūrų: lytis (moteris/vyras), 5 svorio lygiai, mastelis pagal ūgį
- Drabužiai „pasiūti“ ant 3D kūno kiekvienai figūrai, spalva uždedama naršyklėje
- Drabužių uždėjimas ant avataro vienu paspaudimu, nuėmimas, „Nurengti viską“

## Kas bus toliau
- Derinių išsaugojimas ir peržiūra
- Savų drabužių nuotraukų įkėlimas
- DI stilistas

## Kur keisti drabužius
`index.html` faile, skiltyje „1. DUOMENYS“ – kiekviena eilutė `CATALOG` sąraše yra vienas drabužis.

## Avataro paveikslėliai
`assets/avatar/<figūra>/` – `body.png` ir po vieną pilką sluoksnį kiekvienam drabužio tipui. Sugeneruota skriptu `tools/mh_wardrobe.py` iš MakeHuman (CC0) duomenų. Norint pridėti naują drabužio tipą, jį reikia aprašyti skripte ir sugeneruoti iš naujo.
