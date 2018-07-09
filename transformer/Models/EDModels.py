from transformer.Models.Models import *

class EDModels(torchsetModels):
    def __init__(self, dataset,train_loader, test_loader, encoder,decoder):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.encoder = encoder
        self.decoder = decoder
        self.dataset = dataset

    def train(self,input_tensor, target_tensor,encoder_optimizer, decoder_optimizer,criterion, max_length=MAX_LENGTH):
        encoder_hidden = self.encoder.initHidden()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        encoder_outputs = torch.zeros(max_length, self.encoder.hidden_size, device=device)
        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                # print(decoder_output.shape, target_tensor[di])
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                # print(decoder_output, target_tensor[di])
                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_token:
                    break

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        return loss.item() / target_length

    def trainIters(self,n_iters, print_every=100, evaluate_every = 100,save_every=100, learning_rate=0.0001):

        start = time.time()
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)

        criterion = nn.CrossEntropyLoss()
        training_sessions = [random.choice(self.dataset) for i in range(n_iters)]

        for iter in range(1, n_iters + 1):
            session = training_sessions[iter - 1]
            loss = self.train(session[0],session[1], encoder_optimizer,decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))


            if iter % save_every == 0:
                torch.save(self.encoder.state_dict(), 'train_fruit/' + description + '_encoder.pkl')
                torch.save(self.decoder.state_dict(), 'train_fruit/' + description + '_decoder.pkl')

    def trainEpoch(self,epoch_times = 1, print_every=10000, plot_every=100,save_every=100,learning_rate=0.0001):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)

        criterion = nn.CrossEntropyLoss()

        for epoch in range(epoch_times):
            n_iters = len(self.dataset)
            for iter,session in enumerate(self.dataset):
                loss = self.train(session[0],session[1],encoder_optimizer,decoder_optimizer,criterion)
                print_loss_total += loss
                plot_loss_total += loss
                if (iter+epoch*n_iters)%print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    print('%s epoch(%d / %d) iters(%d / %d) %d%% loss is %.4f' % (timeSince(start, ((iter+1)+epoch*n_iters)/ (epoch_times*n_iters)),epoch,epoch_times,
                                                 iter,n_iters ,((iter+1)+epoch*n_iters)/ (epoch_times*n_iters) * 100, print_loss_avg))

                if (iter+epoch*n_iters)% plot_every == 0:
                    plot_loss_avg = plot_loss_total / plot_every
                    plot_losses.append(plot_loss_avg)
                    plot_loss_total = 0
                if (iter + epoch * n_iters) % save_every == 0:
                    torch.save(self.encoder.state_dict(), 'train_fruit/' + description + '_encoder.pkl')
                    torch.save(self.decoder.state_dict(), 'train_fruit/' + description + '_decoder.pkl')

    def evaluate(self, sentence, max_length=MAX_LENGTH):
        with torch.no_grad():
            input_tensor = self.dataset.tensorFromSentence(sentence)
            input_length = len(input_tensor)
            encoder_hidden = self.encoder.initHidden()
            encoder_outputs = torch.zeros(max_length, self.encoder.hidden_size, device=device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei],
                                                         encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

            decoder_hidden = encoder_hidden

            decoded_words = []
            decoder_attentions = torch.zeros(max_length, max_length)

            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(self.dataset.lang.index2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            return decoded_words

    def evaluateRandomly(self,n):
        for i in range(n):
            session = random.choice(self.dataset.unitset)
            session = session.context
            print('>', session[0])
            print('=', session[-1])
            output_words = self.evaluate(session[0],MAX_LENGTH)
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')