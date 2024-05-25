*CategoryModelV1 is trained to classify images into only 7 classes: Animals/Cars/Clothes/Furniture/Jewellery/IT/Other
The IT dataset contains only tv/laptop/camera/tv/printer
The Jewellery dataset contains only earrings / rings / necklace

CategoryModelV1 is the mobilenetv2 finetuned
10 epochs
adam optimizer
Metrics:
accuracy:
0.9352582 overall
0.9777778 for IT class 
1.0 for Clothes
0.997449for Cars
0.8592233for Books
0.98136646 for jewellery
0.903481 for animals
0.789653 for others


CategoryModelV2 is the mobilenetv2 finetuned using 6 classes and threshold for other classes
10 epochs
adam optimizer
Metrics:
accuracy:
0.9737 overall
threshold = 0.95
