
| Model       | Test set    | Accuracy |
| ----------- | ----------- |----------- |
| Baseline    | No transformation   | 0.8259
| Bilateral Blur   | Bilateral Blur        | 0.7404
| Bilateral Blur   | No transformation        | 0.5012
| Contrast | Contrast | 0.7986
| Contrast | No transformation | 0.6517
| random rotate | No transformation | 0.8614
| data augmentation (No transformation, contrast)| No transformation | 0.7849
| data augmentation (No transformation, contrast)| Contrast | 0.7233
| style transfer CT (2000 train) | Style transfer | 0.5150
| style transfer CT (full) | Style transfer | TIMEOUT
| Combine 2 CNN (No transformation, rotate) | No transformation + rotate | 0.8013
| Combine 2 CNN (No transformation, blur) | No transformation + blur | TIMEOUT
| Combine 2 CNN (No transformation, contrast) | No transformation + contrast | TIMEOUT