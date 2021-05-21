"""
COMP5623M Coursework on Image Caption Generation


python decoder.py


"""

import torch
import numpy as np

import torch.nn as nn
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image
import matplotlib.pyplot as pyplot
import random




from datasets import Flickr8k_Images, Flickr8k_Features, Flickr8k_Images_comparison
from models import DecoderRNN, EncoderCNN
from utils import *
from config import *

# if false, train model; otherwise try loading model from checkpoint and evaluate
EVAL = True


# reconstruct the captions and vocab, just as in extract_features.py
lines = read_lines(TOKEN_FILE_TRAIN)
image_ids, cleaned_captions = parse_lines(lines)
vocab = build_vocab(cleaned_captions)


# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# initialize the models and set the learning parameters
decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab), NUM_LAYERS).to(device)


if not EVAL:

    # load the features saved from extract_features.py
    print(len(lines))
    features = torch.load('features.pt', map_location=device)
    print("Loaded features", features.shape)

    features = features.repeat_interleave(5, 0)
    print("Duplicated features", features.shape)

    dataset_train = Flickr8k_Features(
        image_ids=image_ids,
        captions=cleaned_captions,
        vocab=vocab,
        features=features,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=64, # change as needed
        shuffle=True,
        num_workers=0, # may need to set to 0
        collate_fn=caption_collate_fn, # explicitly overwrite the collate_fn
    )


    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)

    print(len(image_ids))
    print(len(cleaned_captions))
    print(features.shape)


#########################################################################
#
#        QUESTION 1.3 Training DecoderRNN
# 
#########################################################################

    # TODO write training loop on decoder here
    num_epochs = 5
    for epoch in range(num_epochs):
        epoch_loss = 0
        n = 0

    # for each batch, prepare the targets using this torch.nn.utils.rnn function
        for data in train_loader:
            features_batch, captions, lengths = data
            features_batch = features_batch.cuda()
            captions = captions.cuda()
            optimizer.zero_grad()
            outputs = decoder(features_batch, captions, lengths)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        total_loss = epoch_loss / n
        print(total_loss)






    # save model after training
    decoder_ckpt = torch.save(decoder, "decoder.ckpt")



# if we already trained, and EVAL == True, reload saved model
else:

    data_transform = transforms.Compose([ 
        transforms.Resize(224),     
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),   # using ImageNet norms
                             (0.229, 0.224, 0.225))])


    test_lines = read_lines(TOKEN_FILE_TEST)
    test_image_ids, test_cleaned_captions = parse_lines(test_lines)
    test_image_ids = test_image_ids[::5]
    # load models
    encoder = EncoderCNN().to(device)
    decoder = torch.load("decoder.ckpt").to(device)
    encoder.eval()
    decoder.eval() # generate caption, eval mode to not influence batchnorm



#########################################################################
#
#        QUESTION 2.1 Generating predictions on test data
# 
#########################################################################


    # TODO define decode_caption() function in utils.py
    # predicted_caption = decode_caption(word_ids, vocab)
    test_set = Flickr8k_Images_comparison(
      image_ids = test_image_ids,
      transform = data_transform,
    )

    # batch_size = 1

    test_loader = torch.utils.data.DataLoader(
      test_set,
      batch_size=1,
      shuffle=False,
      num_workers=0,
    )

    sample_ids = []
    for i, data in enumerate(test_loader):
        image, image_id = data
        image = image.cuda()
        feature = encoder(image)
        if test_loader.batch_size == 1:
            feature = torch.squeeze(feature).unsqueeze(0)
        else:
            feature = torch.squeeze(feature)
        sample_id = decoder.sample(feature)
        sample_ids.append(sample_id)
        # print(sample_ids)
    predicted_caption_batch = decode_caption(sample_ids, vocab)

    print(len(predicted_caption_batch))
    # Store predicted_caption according to image_id
    predicted_caption = dict(zip(test_image_ids, predicted_caption_batch))
    # print(predicted_caption)


    
    img_id = test_image_ids[108]
    print(predicted_caption[img_id])
    print(img_id)
      

   
        
        
    

    


#########################################################################
#
#        QUESTION 2.2-3 Caption evaluation via text similarity 
# 
#########################################################################


    # Feel free to add helper functions to utils.py as needed,
    # documenting what they do in the code and in your report
    total_score = []
    for ids in test_image_ids:
        predicted_captions = []
        
        predicted_captions.append(predicted_caption[ids])
        for sentence in predicted_captions:
            predicted_words = [x for x in sentence.split(' ')]
        
        reference_words = getreference(ids)
        # weights = (1,0,0,0)
        # weights = (0.5,0.5,0,0)
        # weights = (0.33,0.33,0.33,0)
        weights = (0.25,0.25,0.25,0.25)
        bleuscore = BleuScore(reference_words, predicted_words, weights)
        
        total_score.append(bleuscore)
    # print(total_score)
   
    print("BLEU Average: ", avg_score)
  


    
    
    total_cosine_scores = []
    for ids in test_image_ids:
        cos_reference_words = getreference(ids)
        
        cosine_score = Cosine_sim(predicted_caption, cos_reference_words, vocab, ids)
        total_cosine_scores.append(cosine_score)
    final_total_scores = []
    for score in total_cosine_scores:
        cosine_final_score = (score - min(total_cosine_scores)) / (max(total_cosine_scores) - min(total_cosine_scores))
        final_total_scores.append(cosine_final_score)
        

    avg_cosine_score = sum(final_total_scores)/len(final_total_scores)
    
    print("Average Cosine Similarity Score: ", avg_cosine_score)