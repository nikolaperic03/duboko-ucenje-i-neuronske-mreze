# duboko-ucenje-i-neuronske-mreze
# Projekat iz predmeta duboko ucenje i neuronske mreze
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://nbviewer.org/github/nikolaperic03/duboko-ucenje-i-neuronske-mreze/blob/main/Neuronske.ipynb)
## 1. Opis problema
Potrebno je napraviti neuronsku mrežu, koja će omogućiti klasifikaciju email poruka u dve klase (Spam i Ham)
Kao tehnologija za modelovanje neuronske mreže, korišćen PyTorch, dok je za enkodiranje poruka korišćen pretrenirani transformer SBERT, konkretno model _all-MiniLM-L12-v2_. 
Fokus je na kreiranju modela koji minimizuje propuštanje nepoželjnih poruka, obezbeđujući stabilnost i robusnost u realnim uslovima korišćenja.
## 2. Podaci
Korišćeni su podaci, odnosno csv fajl preuzet sa Kaggle-a, [LINK](https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification?resource=download). U Dataset-u nalazi se 5573 instance, pri čemu je dominantna negativna klasa (ham). Prvo je primećeno da postoji izvestan broj duplikata, koji su odmah izbačeni. S obzirom na to da u preuzetom dataset-u dominira klasa Ham, kako dobijeni model ne bi bio pristrasan toj klasi koja dominira, izvršen je downsampling. Nakon downsamplinga, u dataset-u ostalo je 1282 instance, pri čemu su klase potpuno izbalansirane. Dodata je još jedna kolona (numerička) koja predstavlja dužinu teksta u poruci. Analizom grafika utvrđeno je da je dužina teksta značajan prediktor izlazne klase, odnosno da se dešava da mejlovi koji imaju veću dužinu, mogu predstavljati pretnju, u smislu da pripadaju klasi Spam. Sada je potrebno tekst pretvoriti u vektore, tu je korišćen gore pomenuti model, koji od svake rečenice formira vektor dužine (384x1). Nakon enkodiranja tekstualne kolone, skalirana je i numerička kolona koja se odnosi na dužinu teksta poruke. Takvi podaci podeljeni su u trening i test deo, pri čemu je trening deo dodatno podeljen na trening i validacioni deo. Napravljeni su i DataLoader-i za ucitavanje podataka, sa batch size-om od 64
## 3. Arhitektura modela
Model je implementiran kroz klasu "FeedForwardNetDropout", kao višeslojna potpuno povezana neuronska mreža:

1. Ulazni sloj: `nn.Linear(385, 256)` – Prima 385 karakteristika, mapira ih u 256 neurona. Koristi se ReLU aktivaciona funkcija uz Dropout (0.2).
2. Skriveni sloj 1: `nn.Linear(256, 128)` – 128 neurona, uz ReLU aktivacionu funkciju i Dropout (0.2).
3. Skriveni sloj 2 : `nn.Linear(128, 64)` – 64 neurona, uz ReLU aktivacionu funkciju i Dropout (0.2).
4. Izlazni sloj : `nn.Linear(64, 2)` – Vraća sirove vrednosti koje se kasnije prosleđuju funkciji greške.


Uvođenje  Dropout slojeva od 20% ima  ulogu u sprečavanju modela da razvije preveliku zavisnost od specifičnih neurona tokom treninga. Odnosno izbegava se overfitting modela nad podacima.

## 4. Trening
Mreža je trenirana primenom Adam optimizatora i standardne funkcije gubitka za višeklasnu i binarnu klasifikaciju – CrossEntropyLoss.


Korišćena je metoda Early Stopping, kako bi se izbegao overfitting modela nad trening podacima.


Tokom procesa razvoja uočeno je da standardna brzina učenja (Learning Rate) od $0.001$ dovodi do stagnacije modela. Funkcija gubitka se konstatno zadržavala na vriednosti od oko $0.69$), jer je optimizator pravio prevelike korake i stalno preskakao globalni minimum. 

Trening je stabilizovan smanjenjem početne brzine učenja na finiju vrijednost od $lr = 0.0001$, što je omogućilo stabilan i postepen pad greške kroz epohe.


## 5. Analiza osetljivosti i hiperparametrska optimizacija
Implementiran je mehanizam (Early Stopping) sa parametrima patience=5 (strpljenje od 5 epoha) i delta=0.001. Maksimalni broj epoha postavljen je na 40.

Od 1. do 25. epohe Train Loss i Val Loss su opadale. U 25. epohi, model je ostvario svoj minimum greške na validacionom skupu: Val Loss: 0.1286.
Od 26. epohe pa do 30, greška na trening skupu je nastavila da opada sa $0.0646$ na $0.0400$, ali je greška na validacionom skupu počela da raste pet puta zaredom sa $0.1286$ na $0.1334$. 

Nakon toga Early stopping prekida trening, kako se model ne bi overfittovao i čuva parametre modela.

## 6. Rezultati evaluacije
Nakon treninga model je testiran nad novim skupom od 257 poruka, ostvarujući ukupnu tačnost od $94.55%$.


Matrica konfuzije:


[130   8]


[  6 113]


True Negative (130): 130 normlanih poruka je ispravno prepoznato.
False Positive (8): U samo 8 slučajeva je regularna poruka greškom klasifikovana kao spam.
False Negative (6): Samo 6 malicioznih spam poruka je okarakterisano kao normlano.
True Positive (113): 113 spam poruka je uspešno detektovano.

## 7. Diskusija
Zbog visokog recall-a za klasu spam, model se čini veoma pouzdanim i stabilnim. Sa druge strane 8 promešenih poruka, koje su okarakterisane kao spam, a nisu, nije loš rezultat, iako bi model oprezniji, kada bi taj broj bio veći (manje bi poruka propustio ka spam), korisnici ne vole kada im normalne poruke budu označene kao nepoželjne. Upravo taj balans između precision i recall skorova, čini model pouzdanijim i stabilnijim.

## 8. Zaklučak
Kombinacija naprednih semantičkih reprezentacija teksta (dobijenih preko Sentence Transformer modela) i pažljivo optimizovane višeslojne neuronske mreže pokazala se kao izuzetno moćno rešenje za problem detekcije prevara putem elektronske pošte. 


Dokazano je da pravilno podešavanje hiperparametara, pre svega brzine učenja (lr) i implementacija mehanizama Dropout i Early Stopping, predstavljaju važan korak u razvoju stabilnog modela. Razvijeni model uspešno generalizuje naučene obrasce na novim podacima, ostvarujući visoku tačnost od 94.55% i pokazuje spremnost za praktičnu primjenu.

