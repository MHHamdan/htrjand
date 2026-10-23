# Out-of-distribution demo lines: sources and licences

> **In this repository.** Only the line crops are included (the `ood_6` Poggio Bracciolini crops of the original collection were left out because of an institutional claim on the source images). The full-resolution originals referred to below as `originals/...` are NOT redistributed here; download them from the Wikimedia Commons URLs given for each item (the SHA-1 identifies the exact file). These images are out-of-distribution demo material only: the transcriptions are the authors' readings, not ground truth, and no error rate is ever reported on them.


**Demo / out-of-distribution material. Transcriptions are by the authors. This is not a benchmark:** there is no official ground truth, no split, and no reported metric. Use it only to show qualitative behaviour on handwriting the models never saw.

None of these images comes from IAM, RIMES, Bentham, Washington or Saint Gall. None comes from the collections those benchmarks were built from (the Library of Congress George Washington Papers, UCL Bentham Papers, e-codices Cod. Sang. 562).

Every original was downloaded at full resolution from Wikimedia Commons on 2026-10-17. It is kept unmodified in `originals/`; the SHA-1 matches the Commons `imageinfo`. Line crops are plain rectangular PIL crops (`Image.crop(box)`, box = left, top, right, bottom in original pixels), saved as PNG with no other processing. Crops are generous, so fragments of the neighbouring lines' ascenders and descenders are visible in some of them. Only the main line is transcribed.

Transcription convention: diplomatic, keeping the writer's spelling. `[?]` follows an uncertain word. `[...]` marks a cut-off fragment at a crop edge.

---

## ood_1: Abraham Lincoln, letter to Mr. Brayman (1854). Historical English, 19th-c. cursive, photostat-grey background

- File page: https://commons.wikimedia.org/wiki/File:Abraham_Lincoln_Letter_to_Mr._Brayman_September_23,_1854_-_NARA_-_192847.jpg
- Original: https://upload.wikimedia.org/wikipedia/commons/4/49/Abraham_Lincoln_Letter_to_Mr._Brayman_September_23%2C_1854_-_NARA_-_192847.jpg → `originals/lincoln_brayman_1854.jpg` (2351×3000, SHA-1 bd01422a72c25dd1717b26a1474cbbe9a77f43ed)
- Author / date: Abraham Lincoln (1809-1865), Bloomington, 23 Sept 1854. Holding: U.S. National Archives and Records Administration, NAID 192847.
- Licence: **Public domain.** The author died in 1865 and the work was created in 1854. Commons tags: `{{PD-USGov}}`, `{{NARA-cooperation}}`.
- Attribution (not required): "Abraham Lincoln, letter to M. Brayman, 23 Sept. 1854. U.S. National Archives (NAID 192847), via Wikimedia Commons. Public domain."

| crop | box | transcription |
|---|---|---|
| `ood_1_lincoln_a.png` | 150,895,2330,1002 | have negociated for me — I have directed them to |
| `ood_1_lincoln_b.png` | 150,1185,2330,1295 | The reason I have taken this liberty is, that since |

## ood_2: Abraham Lincoln, Gettysburg Address, "Bliss copy" (1864). Clean historical English cursive on ruled paper

- File page: https://commons.wikimedia.org/wiki/File:Gettysburg_Address_Bliss_copy.jpg
- Original: https://upload.wikimedia.org/wikipedia/commons/9/9b/Gettysburg_Address_Bliss_copy.jpg → `originals/lincoln_gettysburg_bliss_1864.jpg` (3827×1588, three pages side by side, SHA-1 ae45675872fb9b69f623a3c033e15ac78878187b)
- Author / date: Abraham Lincoln, fair copy written March 1864.
- Licence: **Public domain** (`{{PD-US}}`; the author died in 1865).
- Attribution (not required): "Abraham Lincoln, Gettysburg Address (Bliss copy), 1864, via Wikimedia Commons. Public domain."

| crop | box | transcription |
|---|---|---|
| `ood_2_gettysburg_a.png` | 120,422,1200,505 | Four score and seven years ago our fathers |
| `ood_2_gettysburg_b.png` | 120,822,1200,905 | testing whether that nation, or any nation |

## ood_3: Charles Darwin, letter to Julius Frazelle Galbraith (11 Nov 1881). Faint, wide-spaced, irregular cursive in grey ink

- File page: https://commons.wikimedia.org/wiki/File:Letter_-_Charles_Darwin_to_Julius_Frazelle_Galbraith_-_1881-11-11.jpg
- Original: https://upload.wikimedia.org/wikipedia/commons/0/04/Letter_-_Charles_Darwin_to_Julius_Frazelle_Galbraith_-_1881-11-11.jpg → `originals/darwin_galbraith_1881.jpg` (1290×2000, SHA-1 d8c315cc4c7e5455fa02a7d5a444e848a2a75317)
- Author / date: Charles Darwin (1809-1882), Down, Beckenham, 11 Nov 1881. The photograph comes from an RR Auction lot listing.
- Licence: **Public domain** (`{{PD-old-70}}`; the author died in 1882). The image is a faithful photograph of a flat document. Under US law (*Bridgeman v. Corel*) and EU DSM Directive art. 14, that adds no new copyright.
- Attribution (not required): "Charles Darwin, letter to J. F. Galbraith, 11 Nov. 1881, via Wikimedia Commons. Public domain."

| crop | box | transcription |
|---|---|---|
| `ood_3_darwin_a.png` | 100,455,1240,560 | I thank you for your kindness |
| `ood_3_darwin_b.png` | 100,565,1240,648 | in having written me so long |

## ood_4: Honoré de Balzac, letter to the publisher Charles Gosselin (Sept 1833). French, 19th-c. rapid cursive, pale ink on grey paper

- File page: https://commons.wikimedia.org/wiki/File:Lettre_autographe_de_Balzac_%C3%A0_M._Gosselin,_libraire_%C3%A9diteur,_BAL_32.jpg
- Original: https://upload.wikimedia.org/wikipedia/commons/8/8c/Lettre_autographe_de_Balzac_%C3%A0_M._Gosselin%2C_libraire_%C3%A9diteur%2C_BAL_32.jpg → `originals/balzac_gosselin_1833.jpg` (3235×2546, SHA-1 8dd4eb9d93f6e10b41f201fbcf84075d06a8e7e7)
- Author / date: Honoré de Balzac (1799-1850), between 12 and 13 Sept 1833. Holding: Maison de Balzac, Paris (inv. BAL 32). Released by Paris Musées: https://www.parismuseescollections.paris.fr/fr/maison-de-balzac/oeuvres/lettre-autographe-de-balzac-a-m-gosselin-libraire-editeur
- Licence: **CC0 1.0** (Paris Musées open-content release; Commons `{{CC-0}}`). The text has been public domain since 1920.
- Attribution (courtesy): "Honoré de Balzac, lettre autographe à M. Gosselin, 1833. Maison de Balzac / Paris Musées, CC0."

| crop | box | transcription |
|---|---|---|
| `ood_4_balzac_a.png` | 1720,893,3090,985 | qui m'a empêché de faire le privilège [?], [...] |
| `ood_4_balzac_b.png` | 1720,1400,3090,1510 | Mais, Monsieur, il y a, je crois, un moyen |

## ood_5: Marie-Antoinette, last letter to Madame Élisabeth (16 Oct 1793), page 1. French, 18th-c., slanted lines, heavy show-through and stains

- File page: https://commons.wikimedia.org/wiki/File:Lettre_de_Marie-Antoinette_%C3%A0_Madame_%C3%89lisabeth._Page_1_-_Archives_Nationales_-_AE-I-7-8_n%C2%B03.jpg
- Original: https://upload.wikimedia.org/wikipedia/commons/0/0d/Lettre_de_Marie-Antoinette_%C3%A0_Madame_%C3%89lisabeth._Page_1_-_Archives_Nationales_-_AE-I-7-8_n%C2%B03.jpg → `originals/marie_antoinette_1793.jpg` (1358×1605, SHA-1 9165c7341a851999ae00608a4e46906d2e09af8b)
- Author / date: Marie-Antoinette (1755-1793), 1793. Holding: Archives nationales (France), AE/I/7-8 n°3, contributed to Commons by the Archives nationales (Wikimédia France cooperation).
- Licence: **Public domain** (`{{PD-old-100-1923}}`).
- Attribution (courtesy): "Marie-Antoinette, lettre à Madame Élisabeth, 1793. Archives nationales, AE/I/7-8 n°3, via Wikimedia Commons. Public domain."

| crop | box | transcription |
|---|---|---|
| `ood_5_marieantoinette_a.png` | 100,232,1270,318 | quant la consience ne reproche rien; j'ai un profond [?] regret d'abandonner |

(The neighbouring lines "fermeté que lui dans ces derniers moments. je suis calme comme on l'est" and "mes pauvres enfants" are partly visible above and below. They are not transcribed.)

## ood_7: Modern German cursive ("Schreibschrift"), pencil on white (2007)

- File page: https://commons.wikimedia.org/wiki/File:Handschrift_(1).jpg
- Original: https://upload.wikimedia.org/wikipedia/commons/d/d7/Handschrift_%281%29.jpg → `originals/modern_de_handschrift_2007.jpg` (2234×1337, SHA-1 b12260e6d4faf7373bd760a8aedfd45ec78ae506)
- Author / date: de.wikipedia user Prince Kassad (own handwriting), 26 Apr 2007.
- Licence: **Public domain** by the author's dedication (`{{PD-self}}`).
- Attribution (courtesy): "Handwriting sample by Prince Kassad, Wikimedia Commons, public domain."
- Note: German is outside the models' training languages. It tests the script only, not the language model.

| crop | box | transcription |
|---|---|---|
| `ood_7_modern_de_a.png` | 0,25,2180,185 | Dies ist eine Schreibschrift, wie sie heutzutage |

## ood_8: Modern English handwriting, two styles (2006). Joined biro cursive in blue, printed felt-tip in red, on yellow paper

- File page: https://commons.wikimedia.org/wiki/File:Handwriting.png
- Original: https://upload.wikimedia.org/wikipedia/commons/a/a7/Handwriting.png → `originals/modern_en_handwriting_2006.png` (701×580, SHA-1 c34a0b15e99b4eac840c83a134e627b0a86a3196)
- Author / date: en.wikipedia user Max Naylor (own handwriting), uploaded 16 June 2006.
- Licence: **Public domain** by the author's release (`{{PD-user-en|Max Naylor}}`).
- Attribution (courtesy): "Handwriting sample by Max Naylor, Wikimedia Commons, public domain."

| crop | box | transcription |
|---|---|---|
| `ood_8_modern_en_a.png` | 50,45,610,120 | The quick brown fox |
| `ood_8_modern_en_b.png` | 40,410,665,480 | jumps over the lazy dog. |
