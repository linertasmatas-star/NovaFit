# NovaFit

Aprangos derinių kūrimo programėlė – Junior Achievement verslumo konkurso projektas.

## Kaip paleisti
Atsisiųskite visą repozitoriją (Code → Download ZIP), išskleiskite ir du kartus spustelėkite `index.html`. Aplankas `assets` turi būti šalia `index.html`. Nieko diegti nereikia.

## Kas jau veikia (demo v6)
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
