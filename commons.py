import io
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import sys
sys.path.append('pytorch-image-models')
import timm

class CFG:
    debug = False
    print_freq = 400
    num_workers = 4
    size = 224
    embed_size = 512
    hidden_size = 512
    vocab_size = 7729
    num_layers = 2
    model_name='resnet10t'
    scheduler='CosineAnnealingLR'
    criterion='CrossEntropyLoss'
    dropout = 0.5
    epochs = 20
    T_max = 3 
    lr = 1e-4
    min_lr = 1e-6
    batch_size = 32
    weight_decay = 1e-6
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 42
    train_flag = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=True):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.backbone = timm.create_model(model_name = CFG.model_name, pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embed_size)
        self.times = []

    def forward(self, images):
        features = self.backbone(images)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(CFG.dropout)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length, count):
        result_caption = []
        all_preds = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for c in range(int(count)):

                for _ in range(max_length):
                    hiddens, states = self.decoderRNN.lstm(x, states)
                    output = self.decoderRNN.linear(hiddens.squeeze(0))

                    val, idx = torch.topk(output, k = int(count), dim = 1)
                    #print(output) # (1, 2994)
                    predicted = idx[0][c]
                    #print(predicted)
                    
                    # predicted = output.argmax(1)
                    # print(predicted)
                    result_caption.append(predicted.item())
                    x = self.decoderRNN.embed( predicted .unsqueeze(0)).unsqueeze(0)
                    
                    if vocabulary[predicted.item()] == "<EOS>":
                        break
                all_preds.append(result_caption)
        
        for i, sentence in enumerate(all_preds):
            temp = [vocabulary[idx] for idx in sentence]
            all_preds[i] = temp
        
        #print(all_preds[1])
        return all_preds
        #return [vocabulary[idx] for idx in result_caption]

def get_model():
	model = CNNtoRNN(CFG.embed_size, CFG.hidden_size, CFG.vocab_size, CFG.num_layers)
	weights_path = 'Loss2.3139_epoch20.bin'
	model.load_state_dict(torch.load(weights_path, map_location = CFG.device), strict=True)
	return model

def get_tensor(image_bytes):
	my_transforms = transforms.Compose([
                    transforms.Resize(224),
        				    transforms.ToTensor(),
        				    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             					  std=[0.229, 0.224, 0.225])])

	image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
	image.save("static/model_photos/original.png")
	return my_transforms(image).unsqueeze(0)

