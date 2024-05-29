Jewellery Classification Model:

First Try:
use the mobilenetv2 model finetuned :
10 epochs
metrics:
0.9837133 overall accuracy
0.9787234 accuracy for earring 
0.98507464 accuracy for necklace
1.0 for ring
with testing set: 184 earring , 61 necklace , 39 ring


Material Classification Model:

First Try:

use the mobilenetv2 model finetuned :
10 epochs
metrics:
0.93670887 overall accuracy
0.98245615 accuracy for gold 
0.85714287 accuracy for silver
0.75 for bronze
with testing set: 57 gold , 14 silver , 8 bronze

Second Try:

use the mobilenetv2 model finetuned with more data and using dataaugmentation :
10 epochs
metrics:
0.9493671 overall accuracy
0.9824561 accuracy for gold 
0.9285714 accuracy for silver
0.75 accuracy for bronze
with testing set: 57 gold , 14 silver , 8 bronze
