import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

from multiprocessing import cpu_count
import numpy as np
import argparse
import os
import time

from WESS import WESS
from loss import WESSLoss
from data_utils import WESSDataLoader, collate_fn, DataLoader
import hparams as hp

if_parallel = False


def main(args):
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cpu")

    # Define model
    model = nn.DataParallel(WESS()).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_bert = BertModel.from_pretrained('bert-base-uncased')
    print("Models Have Been Defined")

    # Get dataset
    dataset = WESSDataLoader(tokenizer, model_bert)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)
    wess_loss = WESSLoss().to(device)
    # loss_list = list()

    # Get training loader
    print("Get Training Loader")
    training_loader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True,
                                 collate_fn=collate_fn, drop_last=True, num_workers=cpu_count())

    # Load checkpoint if exists
    try:
        checkpoint = torch.load(os.path.join(
            hp.checkpoint_path, 'checkpoint_%d.pth.tar' % args.restore_step))
        model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("---Model Restored at Step %d---\n" % args.restore_step)

    except:
        print("---Start New Training---\n")
        if not os.path.exists(hp.checkpoint_path):
            os.mkdir(hp.checkpoint_path)

    # Init logger
    if not os.path.exists("logger"):
        os.mkdir("logger")

    # Training
    model = model.train()

    total_step = hp.epochs * len(training_loader)
    Time = np.array(list())
    Start = time.clock()
    for epoch in range(hp.epochs):
        for i, data_of_batch in enumerate(training_loader):
            start_time = time.clock()

            current_step = i + args.restore_step + \
                epoch * len(training_loader) + 1

            # Init
            optimizer.zero_grad()

            # Prepare Data
            indexs_list = torch.Tensor(
                [i for i in range(hp.batch_size)]).int().to(device)
            # print(indexs_list)

            texts = data_of_batch["text"]
            mels = data_of_batch["mel"]
            embeddings = data_of_batch["embeddings"]
            sep_lists = data_of_batch["sep"]
            gates = data_of_batch["gate"]

            if torch.cuda.is_available():
                texts = torch.from_numpy(texts).long().to(device)
            else:
                texts = torch.from_numpy(texts).long().to(device)

            mels = torch.from_numpy(mels).to(device)
            gates = torch.from_numpy(gates).float().to(device)

            # print("mels:", mels.size())
            # print("gates:", gates.size())
            # print(gates)

            # Forward
            output, mel_target, gate_target = model(
                texts, embeddings, sep_lists, indexs_list, mels, gates)
            mel_output, mel_out_postnet, gate_predicted = output

            # # Test
            # print(mel_out_postnet.size())
            # print(mel_out_postnet)

            # test_mel = mel_out_postnet[0].cpu().detach().numpy()
            # np.save("test_mel.npy", test_mel)

            # print(gate_predicted)

            # print()
            # print("mel target size:", mels.size())
            # print("mel output size:", mel_output.size())
            # print("gate predict:", gate_predicted.size())

            # Calculate loss
            if if_parallel:
                total_loss, mel_loss, gate_loss = wess_loss(
                    mel_output, mel_out_postnet, gate_predicted, mel_target, gate_target)
                # print(gate_loss)
                # loss_list.append(total_loss.item())
                # print(total_loss.item())
            else:
                # print("there")
                total_loss, mel_loss, gate_loss = wess_loss(
                    mel_output, mel_out_postnet, gate_predicted, mels, gates)

            t_l = total_loss.item()
            m_l = mel_loss.item()
            g_l = gate_loss.item()

            with open(os.path.join("logger", "total_loss.txt"), "a") as f_total_loss:
                f_total_loss.write(str(t_l)+"\n")

            with open(os.path.join("logger", "mel_loss.txt"), "a") as f_mel_loss:
                f_mel_loss.write(str(m_l)+"\n")

            with open(os.path.join("logger", "gate_loss.txt"), "a") as f_gate_loss:
                f_gate_loss.write(str(g_l)+"\n")

            # Backward
            total_loss.backward()

            # Clipping gradients to avoid gradient explosion
            nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_thresh)

            # Update weights
            optimizer.step()

            if current_step % hp.log_step == 0:
                Now = time.clock()

                str1 = "Epoch [{}/{}], Step [{}/{}], Gate Loss: {:.4f}, Mel Loss: {:.4f}, Total Loss: {:.4f}.".format(
                    epoch+1, hp.epochs, current_step, total_step, gate_loss.item(), mel_loss.item(), total_loss.item())
                str2 = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                    (Now-Start), (total_step-current_step)*np.mean(Time))

                print(str1)
                print(str2)

                with open(os.path.join("logger", "logger.txt"), "a") as f_logger:
                    f_logger.write(str1 + "\n")
                    f_logger.write(str2 + "\n")
                    f_logger.write("\n")

            if current_step % hp.save_step == 0:
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                )}, os.path.join(hp.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                print("save model at step %d ..." % current_step)

            if current_step in hp.decay_step:
                optimizer = adjust_learning_rate(optimizer, current_step)

            end_time = time.clock()
            Time = np.append(Time, end_time - start_time)
            if len(Time) == hp.clear_Time:
                temp_value = np.mean(Time)
                Time = np.delete(
                    Time, [i for i in range(len(Time))], axis=None)
                Time = np.append(Time, temp_value)


def adjust_learning_rate(optimizer, step):
    if step == hp.decay_step[0]:
        # if step == 20:
        # print("update")
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005

    elif step == hp.decay_step[1]:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0003

    elif step == hp.decay_step[2]:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    return optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int,
                        help='checkpoint', default=3000)
    args = parser.parse_args()
    main(args)
