*ItModel is trained to classify images into only 3 classes: Laptop/Printer/Smartphone/Tv

Test Set: 42 Laptop / 146 Printer / 192 Smartphone / 125 Tv

ItModelV1 is the mobilenetv2 finetuned
10 epochs
adam optimizer
Metrics:
accuracy:
0.9762376 overall
0.97619045 for Laptop
0.96575344 for Printer
0.984375 Smartphone
0.976 Tv

After adding more data and shuffle it these are the results:
ItModelV1 is the mobilenetv2 finetuned
10 epochs
adam optimizer
Metrics:
accuracy:
0.9643564 overall
0.97619045 for Laptop
0.94520545 for Printer
0.9739583 Smartphone
0.968 Tv

So stick with the first version