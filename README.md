# bert_classifier
A BERT based classifier for sentiment classification. The code in this repository contains the script for training the classifier. For training the BERT backbone can either be kept frozen or fine-tuned by specifying the appropriate argument. 

For training the classfier, run the following command:

```bash
python3 classifier.py --option [freeze/finetune] --epochs NUM_EPOCHS --lr LR --train {train_path} --dev {dev_path} --test {test_path}
```
