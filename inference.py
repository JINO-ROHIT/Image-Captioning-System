from commons import get_model, get_tensor
import torch
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('vocab30k.pickle', 'rb') as f:
    vocab_dict = pickle.load(f)

def get_caption(image_bytes, count):
    cap = []
    model = get_model()
    model.to(device)
    model.eval()

    tensor = get_tensor(image_bytes)
    print(tensor.shape)
    with torch.no_grad():
        all_preds = model.caption_image(image = tensor.to(device), 
                                           vocabulary = vocab_dict, 
                                           max_length = 50, 
                                           count = count)
        for sentence in all_preds:
            temp = " ".join(sentence)
        cap.append(temp)
        # cap = " ".join(model.caption_image(image = tensor.to(device), 
        #                                    vocabulary = vocab_dict, 
        #                                    max_length = 50, 
        #                                    count = count))

    return cap

    