# External surgical datasets — synthetic-data backgrounds & disclosure list

Public surgical-video/image datasets used as **real backgrounds** for synthetic FO
insertion (see the synthetic-data plan: borrow real, possibly out-of-distribution,
procedure frames → insert known foreign objects → author + verify labels).

**Why external backgrounds:** the challenge data is only ~200 videos across 2
procedure families, which caps background diversity. Real frames from *other*
procedures give genuine OOD backgrounds (not hallucinated pixels), while the
inserted objects stay within the closed FO vocabulary (`FO_definitions.txt`).

## Compliance (FRAME rule)

FRAME allows only **public** datasets, **disclosed**, and public by the start of
pre-evaluation (`description.md` §4). All datasets below are non-commercial research
license — fine for a research challenge — but every one used **must be disclosed**.
Keep the *Access status* column current; fill *Date obtained* when downloaded.

- Non-commercial (CC BY-NC / -NC-SA) is acceptable: the rule needs *public + disclosed*, not permissive.
- ShareAlike / NoDerivatives terms only bite if we **redistribute images/derivatives**
  (e.g. publishing a synthetic dataset built on them) — training a model and shipping
  weights is not redistribution of the data.

## Shortlist

| Dataset | Procedure (background gained) | Size | License | Access | Modality fit | Access status | Date obtained |
|---|---|---|---|---|---|---|---|
| DSAD (Dresden) | Lap abdominal, 11 anatomies | 13,195 imgs | **CC BY** | Open — Zenodo / GitLab | ✅ high anatomy diversity | ☐ | |
| AutoLaparo | Lap **hysterectomy** | 21 videos 1080p | CC BY-NC-SA 4.0 | Request form | ✅✅ new procedure (OOD) | ☐ | |
| SAR-RARP50 | Robotic **prostatectomy** (suturing) | 50 clips | CC BY 4.0 *(verify on page)* | UCL RDR / Figshare | ✅ new procedure | ☐ | |
| LapGyn4 | Gynecologic laparoscopy | 55k+ imgs | CC BY-NC | Open — ITEC FTP | ✅ new anatomy | ☐ | |
| Cholec80 / CholecT50 / CholecSeg8k | Lap cholecystectomy | 80/50 vids + 8k seg | CC BY-NC-SA 4.0 | Request form (CAMMA) | ✅ in-domain (lapchole) | ☐ | |
| Endoscapes2023 | Lap cholecystectomy (CVS, seg masks) | 201 videos | CC BY-NC-SA 4.0 | PhysioNet / GitHub | ✅ in-domain + masks | ☐ | |
| HeiChole | Lap cholecystectomy | 33 vids, 22h | CC BY-NC-SA | Synapse (account) | ✅ in-domain | ☐ | |
| GraSP / PSI-AVA | Robotic prostatectomy | > PSI-AVA | ⚠️ not clearly stated | Google Drive (email authors) | ✅ new procedure | ☐ confirm license | |
| **MultiBypass140** | **Roux-en-Y gastric bypass** (new proc, multicentric) | 140 videos | CC BY-NC-SA 4.0 | Request form (CAMMA) | ✅✅ new procedure (OOD); **IAE/bleeding frames = hemostatic-agent lead** | ☐ | |
| **SLAM** | Multi-procedure incl. **abdominal wall hernia repair** | 4,097 clips / 34 procs | CC BY-NC-**ND** 4.0 | Open — Figshare (no reg.) | ✅ OOD + **only public MESH source** | ☐ | |
| CholecTrack20 | Lap cholecystectomy (tool tracking) | 20 videos | CC BY-NC-SA 4.0 | Request form (CAMMA) | ⚠️ in-domain; **FO-crop source** (clips, specimen bag) | ☐ | |
| SSG-VQA | Lap cholecystectomy VQA (scene graphs) | on Cholec80 | CC BY-NC-SA 4.0 | CAMMA / GitHub | ⚠️ in-domain; general **VQA/situs SFT augmentation** (not synthetic pipeline) | ☐ | |

## Excluded (with reason)

| Dataset | Reason |
|---|---|
| **ROBUST-MIS 2019** | Same 30 Heidelberg colorectal procedures the challenge's Batch-1 HeiCo-FOCUS is built from → in-distribution, circular as "OOD". |
| **SurgVU / SurgToolLoc** | da Vinci training on **porcine** models, not in-vivo human abdomen → its own distribution shift. |
| CATARACTS / PitVis / Kvasir | Different modality (cataract/pituitary/GI endoscopy) — not laparoscopic abdominal. |
| **MVOR / xawAR16** | External OR-room RGB-D cameras (staff pose / camera tracking) — not the endoscopic surgical field, no foreign objects. |
| **M2CAI 2016** | Subsets *of Cholec80* — redundant. |

## Per-object source map (the 10 FO classes)

8 of 10 FO classes already appear in our own HeiCo + LapChole training data — for
those the pipeline only needs to place them into **OOD backgrounds**. Only **Mesh**
and **Absorbable Hemostatic Agent** are missing and need external sourcing.

| FO class | In our data? | Real-crop source | Strategy |
|---|---|---|---|
| Clip | ✅ | HeiCo/LapChole + Cholec80/T50/Track20, Endoscapes | insert into OOD bg |
| Sponge | ✅ | HeiCo/LapChole | insert into OOD bg |
| Specimen Bag | ✅ | LapChole + Cholec datasets | insert into OOD bg |
| Gallstone | ✅ | LapChole + Cholec datasets | insert into OOD bg |
| Specimen | ✅ | HeiCo/LapChole | insert into OOD bg |
| Needle | ✅ | HeiCo/LapChole + SAR-RARP50, MultiBypass140 | insert into OOD bg |
| Silicone Loop | ✅ | HeiCo (colorectal) | insert into OOD bg |
| External Drain | ✅ | HeiCo/LapChole | insert into OOD bg |
| **Mesh** | ⚠️ 5 samples | **SLAM** (hernia clips → mine w/ SAM2) | cut real + insert |
| **Absorbable Hemostatic Agent** | ❌ 0 samples | **none** (MultiBypass140 bleeding frames = weak seed) | **synthetic-only** (editing-AI) |

## Datasets by role in the pipeline

- **OOD backgrounds (highest value):** AutoLaparo (hysterectomy), SAR-RARP50
  (prostatectomy), MultiBypass140 (gastric bypass), DSAD (broad abdominal), LapGyn4
  (gyn), SLAM (multi-proc incl. hernia).
- **In-domain backgrounds (lower value):** Cholec80/T50/Track20, Endoscapes, HeiChole.
- **Real object-crop sources:** own HeiCo/LapChole (common FOs) + Cholec80/T50
  (triplet/bbox labels locate clips/bags/gallstones); **SLAM** for mesh.
- **No source → synthetic-only:** Absorbable Hemostatic Agent.
- **VQA/situs SFT augmentation (separate from synthetic pipeline):** SSG-VQA.

## Priority

1. **DSAD** — open (no gate), CC BY, broad abdominal anatomy → prototype the
   SAM2-extract → composite → verify loop here first.
2. **AutoLaparo + SAR-RARP50 + MultiBypass140** — genuine unseen-procedure backgrounds
   (hysterectomy, prostatectomy, gastric bypass).
3. **SLAM** — grab the mesh (the one class with a real public source).
4. **Cholec80 / Endoscapes** — in-domain (lapchole) object-insertion practice + FO crops.

## Links

- CAMMA (Cholec80/T50, Endoscapes): https://camma.unistra.fr/datasets/
- Endoscapes: https://github.com/CAMMA-public/Endoscapes
- AutoLaparo: https://autolaparo.github.io/
- DSAD: https://zenodo.org/records/6958337 · https://gitlab.com/nct_tso_public/dsad
- LapGyn4: http://ftp.itec.aau.at/datasets/LapGyn4/
- SAR-RARP50: https://rdr.ucl.ac.uk/projects/SAR-RARP50_Segmentation_of_surgical_instrumentation_and_Action_Recognition_on_Robot-Assisted_Radical_Prostatectomy_Challenge/191091
- GraSP: https://github.com/BCV-Uniandes/GraSP
- HeiChole: https://www.synapse.org/Heichole
- MultiBypass140: https://github.com/CAMMA-public/MultiBypass140
- SLAM (Figshare): https://www.nature.com/articles/s41597-025-05093-7 (data on Figshare, linked from paper)
- CholecTrack20: https://github.com/CAMMA-public/cholectrack20
- SSG-VQA: https://github.com/CAMMA-public/SSG-VQA
