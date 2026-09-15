# Wyniki eksperymentów E1 i E2 — notatka

Streszczenie tego, co pokazuje `notebooks/wyniki.ipynb`. Notatnik niczego nie liczy —
wyświetla pliki `results/e1_toy_metrics.csv` i `results/e2_uci_metrics.csv`, wyprodukowane
przez `experiments/e1_toy.py` i `experiments/e2_uci.py`, przez funkcje z `src/tables.py`
(testy: `tests/test_tables.py`). Wszystkie liczby poniżej są przepisane z jego wyjścia.
Pełne wpisy do zastrzeżeń (`N-xxx`) i decyzji (`DEC-xxx`) są w `docs/experiment_notes.md`.

Porównywanych jest sześć metod: `map` (deterministyczny punkt odniesienia), `mcd`
(MC dropout), `ensemble` (deep ensemble, 5 sieci), `bbb` (Bayes by Backprop), `laplace`
(aproksymacja Laplace'a, pełna sieć, GGN), `gp` (proces gaussowski — jedyna dokładna posterior).

## 0. Konwencje wspólne dla obu etapów

| | E1 — problem syntetyczny | E2 — dane rzeczywiste |
|---|---|---|
| dane | `y = sin(x) + N(0, 0.1²)`, x ~ U[0, 6], N_train = 250 | UCI: yacht, energy, concrete |
| powtórzenia | 20 ziaren (ziarno losuje dane **i** model) | 20 splitów 90/10 Hernándeza-Lobato i Adamsa (2015); ziarno = numer splitu, steruje tylko inicjalizacją i kolejnością batchy |
| podział wyników | 3 regiony: `in_range` [0, 6], `extrapolation` [-2, 0] ∪ [6, 8], `overall` | 1 wiersz na (metoda, zbiór) |
| rysunki | tak (1D) | nie (d = 6–8) |
| prawdziwy szum | znany, σ = 0.1 | nieznany |

Metryki: **RMSE** (mniej = lepiej), **LL** — średnia log-wiarygodność testowa (więcej = lepiej),
**PICP@95** (cel 0.95), **MPIW@95** — nigdy bez PICP, plus diagnostyczne `sigma_fitted`
(dopasowany szum obserwacyjny), `cal_err` (średnie |odchylenie pokrycia od nominału| po wielu
poziomach) i `cal_bias` (to samo ze znakiem: + = za szerokie, − = za wąskie / nadpewność).

Ustalenia, których nie wolno pominąć przy cytowaniu:

- rozrzut `±` to **odchylenie standardowe** (DEC-006, DEC-015), nie błąd standardowy; literatura
  (Hernández-Lobato i Adams, Gal i Ghahramani) podaje SE, przy n = 20 `SE = std / 4.472`;
- metryki w **oryginalnych jednostkach y** (DEC-003) — uczenie na danych standaryzowanych,
  predykcje odstandaryzowane przed liczeniem metryk;
- przedział 95% z `z = 1.959964` (DEC-008); rysunki pokazują `±2σ` (DEC-001) — ~2% różnicy,
  podpis rysunku ma mówić `±2σ`, nie „95%";
- każda metoda oceniana jako **jeden Gauss dopasowany momentami**
  `N(μ, var_aleatoric + var_epistemic)` (DEC-009), także metody próbkujące — porównywalne
  między sobą, ale **nie** z tabelami Gala (mieszanka log-sum-exp).

---

## Część I — problem syntetyczny (E1)

Konfiguracja odczytana z CSV: `n_train = 250`, `grid_points = 500` (300 w zakresie, 200 poza),
`noise_sigma = 0.1`, `epochs = 4000`, 20 ziaren, `torch_threads = 1`, `run_id = 20260905T005453Z`.

### I.1 Region w zakresie treningowym [0, 6]

| metoda | RMSE | LL | PICP@95 | MPIW@95 | sigma_fitted | cal_err | cal_bias |
|---|---|---|---|---|---|---|---|
| map | 0.102 ± 0.006 | 0.857 ± 0.059 | 0.943 ± 0.019 | 0.396 ± 0.016 | 0.1011 ± 0.0042 | 0.0238 ± 0.0100 | −0.0012 ± 0.0240 |
| mcd | 0.154 ± 0.008 | 0.310 ± 0.045 | 0.999 ± 0.003 | 0.949 ± 0.052 | 0.2018 ± 0.0120 | 0.1352 ± 0.0169 | 0.1352 ± 0.0169 |
| ensemble | 0.100 ± 0.005 | 0.875 ± 0.045 | 0.950 ± 0.016 | 0.400 ± 0.017 | 0.1007 ± 0.0043 | 0.0240 ± 0.0113 | 0.0053 ± 0.0242 |
| bbb | 0.102 ± 0.004 | 0.857 ± 0.041 | 0.950 ± 0.015 | 0.406 ± 0.017 | 0.1027 ± 0.0044 | 0.0220 ± 0.0110 | 0.0041 ± 0.0221 |
| laplace | 0.102 ± 0.006 | 0.858 ± 0.057 | 0.947 ± 0.018 | 0.402 ± 0.016 | 0.1011 ± 0.0042 | 0.0236 ± 0.0102 | 0.0035 ± 0.0238 |
| gp | 0.100 ± 0.005 | 0.879 ± 0.045 | 0.947 ± 0.019 | 0.394 ± 0.017 | 0.0993 ± 0.0043 | 0.0229 ± 0.0112 | 0.0028 ± 0.0234 |

**Wynik pozytywny:** w interpolacji pięć z sześciu metod jest nieodróżnialnych — RMSE ≈ 0.10
(poziom szumu, lepiej się nie da), PICP ≈ 0.95, `sigma_fitted` ≈ 0.1. `mcd` odstaje: przedział
2.4× szerszy, PICP 0.999, szum dwa razy za duży (N-001, patrz I.4).

### I.2 Region ekstrapolacji [-2, 0] ∪ [6, 8] — główny wynik E1

| metoda | RMSE | LL | PICP@95 | MPIW@95 | cal_bias |
|---|---|---|---|---|---|
| map | 0.359 ± 0.070 | −5.227 ± 2.710 | 0.585 ± 0.066 | 0.396 ± 0.016 | −0.2546 ± 0.0385 |
| mcd | 0.806 ± 0.077 | −4.186 ± 0.896 | 0.156 ± 0.054 | 1.048 ± 0.054 | −0.5217 ± 0.0204 |
| ensemble | 0.347 ± 0.047 | −3.143 ± 1.240 | 0.622 ± 0.055 | 0.432 ± 0.033 | −0.2302 ± 0.0305 |
| bbb | 0.210 ± 0.048 | −0.780 ± 1.077 | 0.689 ± 0.136 | 0.410 ± 0.018 | −0.2106 ± 0.0866 |
| laplace | 0.359 ± 0.070 | −0.010 ± 0.165 | 0.923 ± 0.057 | 0.924 ± 0.113 | −0.0636 ± 0.0631 |
| gp | 0.267 ± 0.123 | 0.028 ± 0.301 | 0.962 ± 0.064 | 1.144 ± 0.065 | 0.0218 ± 0.1333 |

(`sigma_fitted` jest globalne, więc identyczne jak w I.1.)

- **Kalibrację poza danymi utrzymują tylko `gp` i `laplace`** — poszerzają pasmo, LL zostaje
  blisko zera, PICP 0.96 / 0.92.
- `map` nie ma członu epistemicznego (`var_epistemic ≡ 0`), pasmo się nie zmienia, PICP spada
  do 0.59, LL się załamuje.
- `ensemble` i `bbb` poszerzają pasmo za mało, żeby nadążyć za błędem średniej (PICP 0.62 / 0.69).
- **`mcd` jest najbardziej pouczający**: najszersze pasmo ze wszystkich (MPIW 1.048), a mimo to
  najgorsze pokrycie (0.156) i najgorsze RMSE (0.806). Pasmo jest szerokie *równomiernie* (bo
  bierze się z globalnego szumu), a nie tam, gdzie kończą się dane. To jest dokładnie powód
  zasady „MPIW nigdy bez PICP": po samej szerokości `mcd` wyglądałby na najostrożniejszą metodę,
  a jest najbardziej nadpewny.

Region `overall` jest w notatniku dla kompletności (miesza oba regiony w proporcji 300:200
wynikającej z siatki) — do wniosków cytować dwa powyższe osobno.

### I.3 Kontrola strukturalna — iloraz MPIW ekstrapolacja / zakres

| metoda | MPIW w zakresie | MPIW ekstrapolacja | iloraz |
|---|---|---|---|
| gp | 0.394 | 1.144 | 2.90 |
| laplace | 0.402 | 0.924 | 2.30 |
| mcd | 0.949 | 1.048 | 1.10 |
| ensemble | 0.400 | 0.432 | 1.08 |
| bbb | 0.406 | 0.410 | 1.01 |
| map | 0.396 | 0.396 | 1.00 |

Uporządkowanie zgodne z oczekiwaniem z teorii: `gp` i `laplace` mocno, `ensemble` trochę,
`bbb` ledwo, `map` dokładnie 1.00 (sanity check pipeline'u). `mcd` ma iloraz ≈ 1 z innego
powodu niż `map`: jego pasmo jest szerokie wszędzie.

### I.4 `sigma_fitted` względem znanej prawdy σ = 0.1 (tylko E1)

| metoda | sigma_fitted | iloraz (sd) | iloraz (wariancja) |
|---|---|---|---|
| map | 0.1011 ± 0.0042 | 1.01 | 1.02 |
| mcd | 0.2018 ± 0.0120 | 2.02 | 4.07 |
| ensemble | 0.1007 ± 0.0043 | 1.01 | 1.01 |
| bbb | 0.1027 ± 0.0044 | 1.03 | 1.06 |
| laplace | 0.1011 ± 0.0042 | 1.01 | 1.02 |
| gp | 0.0993 ± 0.0043 | 0.99 | 0.99 |

**N-001:** pięć metod trafia w prawdziwy szum z dokładnością do kilku procent; `mcd`
przeszacowuje go 2× w odchyleniu (~4× w wariancji). Przyczyna strukturalna: model jest
homoskedastyczny (wymusza to `laplace-torch` przyjmujący skalarne `sigma_noise`), więc szum
dropoutu nie ma gdzie pójść poza globalne `log_sigma2` — dropout „przebiera się" za szum
obserwacyjny.

**Zastrzeżenie do tekstu (N-015):** nasze `dropout_p = 0.1` jest 2–20× większe niż 0.05 i 0.005
u Gala na tych samych zbiorach. Zdanie „MC dropout przeszacowuje szum" bez „przy `dropout_p =
0.1`, wartości odziedziczonej i wyższej niż źródłowa" byłoby nieuczciwe wobec metody.

### I.5 Rysunki (`figures/e1_toy_all.png`, `figures/e1_toy_<metoda>.png`, ziarno 0)

Pasmo wewnętrzne = `±2σ` aleatoryczne, zewnętrzne = `±2σ` całości; odstęp między nimi to człon
epistemiczny. Wspólne osie na panelu zbiorczym są istotne — przycięte pasmo odwróciłoby
porównanie.

- `map` — pasmo stałej szerokości na całej dziedzinie, oba pasma się pokrywają.
- `mcd` — pasmo szerokie wszędzie, także tam, gdzie dane są gęste.
- `ensemble` — lekkie rozejście na brzegach, znacznie słabsze niż `gp`/`laplace`.
- `bbb` — pasmo prawie nie reaguje na wyjście poza dane (znane ograniczenie rodziny
  średniopolowej przy jednej warstwie ukrytej).
- `laplace` — pasmo rośnie wyraźnie poza [0, 6]; średnia identyczna jak `map` (ta sama sieć).
- `gp` — pasmo wraca do a priori z dala od danych, wzorcowe zachowanie.

---

## Część II — dane rzeczywiste (E2)

Trzy zbiory UCI na oryginalnych 20 splitach Hernándeza-Lobato i Adamsa (wczytywanych z plików
indeksów). Przetwarzanie: standaryzacja na statystykach treningowych splitu i nic więcej (bez
logarytmów, outlierów, one-hot, selekcji cech — każde z nich zniweczyłoby porównanie
z literaturą, które jest jedyną zewnętrzną walidacją etapu).

| zbiór | N_train | N_test | d | P (parametry sieci) | N_train/P |
|---|---|---|---|---|---|
| yacht | 277 | 31 | 6 | 402 | 0.69 |
| energy | 691 | 77 | 8 | 502 | 1.38 |
| concrete | 927 | 103 | 8 | 502 | 1.85 |

Na `yacht` parametrów jest więcej niż punktów treningowych. GGN ma rangę ≤ N_train, więc 124
kierunki są określone przez a priori — **sprawdzone (N-020): wnoszą 0.0000 wariancji
epistemicznej w punktach testowych**, bo jakobiany testowe nie mają wzdłuż nich składowej.
Nie wolno pisać, że niepewność Laplace'a na `yacht` jest podyktowana a priori. Cały etap siedzi
w reżimie małych danych (N_train/P od 0.69 do 1.85).

**Bez agregatu po zbiorach** (DEC-016) — cele w różnych jednostkach.

### II.1 yacht

| metoda | RMSE | LL | PICP@95 | MPIW@95 | sigma_fitted | cal_err | cal_bias |
|---|---|---|---|---|---|---|---|
| map | 0.562 ± 0.173 | −1.017 ± 0.591 | 0.903 ± 0.056 | 1.684 ± 0.210 | 0.4295 ± 0.0535 | 0.0835 ± 0.0256 | 0.0264 ± 0.0622 |
| mcd | 1.668 ± 0.483 | −2.328 ± 0.086 | 0.987 ± 0.019 | 13.714 ± 0.329 | 2.7320 ± 0.0879 | 0.2502 ± 0.0426 | 0.2455 ± 0.0495 |
| ensemble | 0.492 ± 0.169 | −0.566 ± 0.144 | 0.965 ± 0.033 | 2.167 ± 0.181 | 0.4142 ± 0.0179 | 0.1675 ± 0.0452 | 0.1631 ± 0.0478 |
| bbb | 0.625 ± 0.302 | −1.300 ± 1.212 | 0.884 ± 0.077 | 1.729 ± 0.176 | 0.4367 ± 0.0447 | 0.1043 ± 0.0740 | −0.0145 ± 0.1222 |
| laplace | 0.562 ± 0.173 | −0.679 ± 0.182 | 0.953 ± 0.045 | 2.023 ± 0.208 | 0.4295 ± 0.0535 | 0.0964 ± 0.0415 | 0.0739 ± 0.0571 |
| gp | 0.436 ± 0.149 | −0.148 ± 0.196 | 0.919 ± 0.055 | 1.055 ± 0.220 | 0.1510 ± 0.0131 | 0.0667 ± 0.0246 | −0.0069 ± 0.0610 |

`gp` wygrywa wyraźnie w RMSE i LL (−0.148 wobec −0.566 u następnego). `mcd` najgorszy
z dużym marginesem, pasmo 6× szersze niż reszta (ten sam mechanizm co N-001, silniejszy).
**N-016:** cel `yacht` jest silnie skośny (+1.75), nieujemny, połowa obserwacji w dolnych 5%
zakresu — homoskedastyczny Gauss z jedną globalną sigmą kładzie masę poniżej zera; uderza to
w LL/PICP/MPIW, RMSE tego nie widzi.

### II.2 energy

| metoda | RMSE | LL | PICP@95 | MPIW@95 | sigma_fitted | cal_err | cal_bias |
|---|---|---|---|---|---|---|---|
| map | 0.474 ± 0.076 | −0.804 ± 0.299 | 0.881 ± 0.042 | 1.387 ± 0.070 | 0.3538 ± 0.0178 | 0.0633 ± 0.0293 | −0.0546 ± 0.0396 |
| mcd | 1.165 ± 0.113 | −1.769 ± 0.036 | 0.998 ± 0.005 | 7.668 ± 0.173 | 1.5833 ± 0.0370 | 0.1696 ± 0.0296 | 0.1696 ± 0.0296 |
| ensemble | 0.401 ± 0.059 | −0.517 ± 0.139 | 0.963 ± 0.015 | 1.702 ± 0.065 | 0.3505 ± 0.0074 | 0.0775 ± 0.0261 | 0.0749 ± 0.0278 |
| bbb | 0.524 ± 0.056 | −0.843 ± 0.188 | 0.902 ± 0.031 | 1.678 ± 0.124 | 0.4267 ± 0.0315 | 0.0512 ± 0.0285 | −0.0278 ± 0.0474 |
| laplace | 0.474 ± 0.076 | −0.700 ± 0.217 | 0.920 ± 0.033 | 1.579 ± 0.071 | 0.3538 ± 0.0178 | 0.0424 ± 0.0182 | −0.0157 ± 0.0386 |
| gp | 0.466 ± 0.067 | −0.665 ± 0.165 | 0.923 ± 0.027 | 1.712 ± 0.045 | 0.3848 ± 0.0085 | 0.0541 ± 0.0225 | 0.0377 ± 0.0342 |

`ensemble` najlepszy w RMSE i LL, `laplace` najlepiej skalibrowany (`cal_err` 0.042), `gp`
w środku stawki — **inny porządek niż na `yacht`**.

> **Wiersz `gp` zastrzeżony (N-021):** amplituda jądra `constant_value` na górnej granicy
> `1e5` w **20/20 splitów**, 16/20 dopasowań z ostrzeżeniem o braku zbieżności (`yacht`
> i `concrete` czyste: min. margines ≥ 8.7 jednostki log, zero ostrzeżeń). Diagnostyczne
> przeliczenie z granicą `1e12` (nic z tego w `results/`): RMSE 0.4662 → 0.4751,
> LL −0.6655 → −0.6869, PICP 0.9234 → 0.9266, MPIW 1.7118 → 1.7333, mediana amplitudy
> 1e5 → 2.11e6. Granica naprawdę wiąże, ale predykcje prawie się nie ruszają (i lekko na
> gorsze) — klasyczna nieidentyfikowalność amplituda/skala długości. **Decyzja autora:**
> raportować z zastrzeżeniem, poszerzyć granicę w `src/methods/gp.py` i przeliczyć oba etapy,
> albo zostawić komórkę pustą. Bez wzmianki raportować nie wolno.

**N-016 (drugie):** `energy` ma jedną cechę binarną i trzy czterowartościowe traktowane jak
ciągłe — tak jak w repozytorium źródłowym; one-hot zmieniłby `d` i porównywalność.

### II.3 concrete

| metoda | RMSE | LL | PICP@95 | MPIW@95 | sigma_fitted | cal_err | cal_bias |
|---|---|---|---|---|---|---|---|
| map | 4.958 ± 0.475 | −3.615 ± 0.351 | 0.812 ± 0.033 | 10.640 ± 0.586 | 2.7143 ± 0.1495 | 0.0962 ± 0.0333 | −0.0928 ± 0.0375 |
| mcd | 5.611 ± 0.385 | −3.159 ± 0.053 | 0.965 ± 0.015 | 24.654 ± 0.387 | 5.6167 ± 0.1123 | 0.0679 ± 0.0245 | 0.0658 ± 0.0254 |
| ensemble | 4.352 ± 0.657 | −2.861 ± 0.211 | 0.920 ± 0.031 | 13.902 ± 0.544 | 2.7161 ± 0.0883 | 0.0479 ± 0.0193 | 0.0287 ± 0.0306 |
| bbb | 5.279 ± 0.703 | −3.667 ± 0.452 | 0.809 ± 0.041 | 11.471 ± 0.827 | 2.9193 ± 0.2106 | 0.1147 ± 0.0320 | −0.1140 ± 0.0326 |
| laplace | 4.958 ± 0.475 | −3.082 ± 0.218 | 0.871 ± 0.036 | 13.042 ± 0.650 | 2.7143 ± 0.1495 | 0.0536 ± 0.0213 | −0.0384 ± 0.0347 |
| gp | 5.406 ± 0.466 | −3.065 ± 0.088 | 0.937 ± 0.024 | 19.875 ± 0.399 | 4.3809 ± 0.0900 | 0.0391 ± 0.0187 | 0.0230 ± 0.0273 |

**Najciekawszy wynik negatywny etapu:** `gp` ma najgorsze RMSE z sześciu metod (5.406 wobec
4.352 u `ensemble`) i jednocześnie najlepszy `cal_err`. Dokładna posterior nie jest
automatycznie najlepszą metodą na prawdziwych danych — obraz z E1 („gp wzorcem pod każdym
względem") **nie przenosi się**.

**`map` vs `laplace`:** identyczne RMSE i `sigma_fitted` na wszystkich trzech zbiorach (średnia
Laplace'a *jest* siecią MAP), różnią się tylko LL/PICP/MPIW/kalibracją. Na `concrete` LL −3.615
→ −3.082 przy zerowej zmianie RMSE — najczystsza demonstracja tego, co daje dołożenie aparatu
niepewności do gotowego modelu.

**N-018:** `concrete` ma 25 zduplikowanych wierszy → 70 z 2060 wierszy testowych (3.4%)
powtarza wiersz treningowy. Własność danych, na których uczył się Gal; porównanie między
metodami nienaruszone, bezwzględny poziom liczb lekko optymistyczny.

### II.4 Kalibracja — PICP@95 na trzech zbiorach

| metoda | yacht | energy | concrete |
|---|---|---|---|
| map | 0.903 | 0.881 | 0.812 |
| mcd | 0.987 | 0.998 | 0.965 |
| ensemble | 0.965 | 0.963 | 0.920 |
| bbb | 0.884 | 0.902 | 0.809 |
| laplace | 0.953 | 0.920 | 0.871 |
| gp | 0.919 | 0.923 | 0.937 |

**Nic nie jest dobrze skalibrowane inaczej niż przypadkiem** — zakres 0.809–0.998. `map` i `bbb`
na `concrete` ~0.81 (nadpewność), `mcd` przekracza nominał wszędzie (przedział absurdalnie
szeroki). Do postawienia w pracy wprost: na danych rzeczywistych żadna metoda nie daje
przedziału o zadeklarowanym pokryciu, a kierunek błędu różni się między metodami.

### II.5 `sigma_fitted` — N-001 na danych rzeczywistych (porównanie względne)

| zbiór | mcd | średnia pozostałych pięciu | iloraz (sd) | iloraz (wariancja) |
|---|---|---|---|---|
| yacht | 2.7320 | 0.3722 | 7.34 | 53.9 |
| energy | 1.5833 | 0.3739 | 4.23 | 17.9 |
| concrete | 5.6167 | 3.0890 | 1.82 | 3.3 |

N-001 potwierdza się, a jego siła zależy od zbioru: zawyżenie tym większe, im mniej prawdziwego
szumu ma zbiór. `yacht` to niemal deterministyczny eksperyment (GP dopasowuje `noise_level`
~1e-4), więc szum dropoutu wychodzi ~54× w wariancji; `concrete` ma realny szum i zawyżenie
niemal znika (3.3×). W E1 było 4.07× wobec znanej prawdy.

### II.6 Porównanie z literaturą — kontrola pipeline'u, nie wynik

Nasze `mcd` vs `yaringal/DropoutUncertaintyExps` (1 warstwa ukryta, 40×100 epok, te same
splity). Literatura przeliczona z SE na std (`std = SE · √20`), żeby obie strony były na tej
samej konwencji.

| zbiór | RMSE (nasze) | RMSE (literatura) | LL (nasze) | LL (literatura) |
|---|---|---|---|---|
| yacht | 1.668 ± 0.483 | 0.666 ± 0.223 | −2.328 ± 0.086 | −1.250 ± 0.070 |
| energy | 1.165 ± 0.113 | 0.539 ± 0.062 | −1.769 ± 0.036 | −1.212 ± 0.024 |
| concrete | 5.611 ± 0.385 | 4.826 ± 0.737 | −3.159 ± 0.053 | −2.937 ± 0.114 |

Kryterium: ten sam rząd wielkości, nie równość — spełnione. Jesteśmy gorsi wszędzie (2.5×,
2.2×, 1.16× w RMSE), najbardziej tam, gdzie największe zawyżenie z II.5. Cztery znane,
niezamknięte rozbieżności: (1) `dropout_p = 0.1` wobec strojonej siatki Gala (0.05 / 0.005);
(2) kara z `gamma` wobec L2 z `tau`; (3) nasz pojedynczy Gauss (DEC-009) wobec mieszanki
log-sum-exp; (4) ich RMSE uśrednione po przebiegach stochastycznych. Tabela waliduje
pipeline, nie naszą konfigurację `mcd`.

---

## Lista zastrzeżeń (skrót; pełne wpisy w `docs/experiment_notes.md`)

| # | czego dotyczy | status |
|---|---|---|
| N-001 / N-015 | `mcd` zawyża szum, ale przy `dropout_p = 0.1` — 2–20× więcej niż Gal | otwarte, do opisania w tekście |
| N-016 | `yacht`: skośny, nieujemny cel vs homoskedastyczny Gauss; `energy`: cechy kategoryczne jako ciągłe | ograniczenia do zaraportowania |
| N-018 | `concrete`: 25 duplikatów, 3.4% testu powtarza trening | własność danych, opisana |
| N-019 | `curvlinops` liczy krzywiznę Laplace'a w float32; na `yacht` wskaźnik uwarunkowania 7.5e6 (E1: 1.26e5); zmierzony błąd 0.017% szerokości pasma | decyzja autora |
| N-020 | deficyt rangi na `yacht` realny, ale nie przenosi się na pasmo (przewidywanie z N-013 zmierzone, nie potwierdziło się) | zamknięte pomiarem |
| N-021 | `energy`, `gp`: amplituda na granicy 20/20; wpływ na metryki < 2% | decyzja autora |
| DEC-016 | brak agregatu po zbiorach | ustalone |

## Wnioski do rozdziału — jednym akapitem

W interpolacji (E1, [0, 6]) pięć z sześciu metod jest równoważnych i dobrze skalibrowanych;
różnice pojawiają się dopiero poza danymi, gdzie kalibrację utrzymują tylko `gp` i `laplace`,
a `mcd` pokazuje, że szerokie pasmo nie jest ostrożnością, jeśli jest szerokie w złym miejscu.
Na danych rzeczywistych (E2) ranking zmienia się ze zbioru na zbiór (`gp` wygrywa na `yacht`,
`ensemble` na `energy` i `concrete`; `gp` na `concrete` ma najgorsze RMSE przy najlepszej
kalibracji), żadna metoda nie osiąga nominalnego pokrycia 0.95, a para `map`/`laplace`
pokazuje w czystej postaci, co daje dołożenie niepewności bez zmiany średniej. Zawyżanie szumu
przez MC dropout (N-001) jest systematyczne i tym silniejsze, im mniej szumu ma zbiór — ale
musi być raportowane razem z wartością `dropout_p`.
