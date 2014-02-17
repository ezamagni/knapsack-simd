-------------------------------------------------------------------------------
Implementazione e studio di un algoritmo genetico 0-1 Knapsack su GPGPU
Attività progettuale per il corso di: FONDAMENTI DI INTELLIGENZA ARTIFICIALE M
a cura di: Enrico Zamagni
Anno accademico: 2011/12
Prof. Andrea Roli
Prof.ssa Paola Mello
-------------------------------------------------------------------------------


La cartella Knapsack_GA_CPU contiene i sorgenti dell'algoritmo sviluppato
funzionante su CPU tradizionale e l'euristica casuale utilizzata come
confronto e verifica. E' possibile compilare entrambi i binari lanciando
l'utility make all'interno della cartella.

La cartella Knapsack_GA_GPU contiene la versione dell'algoritmo realizzata
per il framework CUDA. Per compilarla è sufficiente lanciare l'utility make
all'interno della cartella.

La cartella instances dispone di alcune istanze di esempio da utilizzare come
test per gli algoritmi sopra compilati.

Per generare ulteriori istanze di 0-1 Knapsack è possibile utilizzare il
generatore realizzato da D. Pisinger, S. Martello e P. Toth  
(http://www.diku.dk/hjemmesider/ansatte/pisinger/codes.html) presente nella 
cartella generator.


Bologna,
17 Aprile 2012
Enrico Zamagni