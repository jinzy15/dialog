from transformer.Models.HRED import *

class AHRED(HRED):

    # for hred, train should take the context of the previous turn
    # should return current loss as well as context representation

    def train(self, session, len_session, criterion, encoder_optimizer, context_optimizer, decoder_optimizer):
        # print len(training_group)
        context = self.context
        encoder = self.encoder
        decoder = self.decoder

        context_hidden = context.initHidden()
        context_optimizer.zero_grad()
        encoder_optimizer.zero_grad()  # pytorch accumulates gradients, so zero grad clears them up.
        decoder_optimizer.zero_grad()

        for i in range(0, len_session - 1):
            input_variable = session[i]
            target_variable = session[i + 1]
            last = False
            if i + 1 == len_session - 1:
                last = True

            if last:
                loss, context_hidden = self.trainApair(input_variable, target_variable, encoder,
                                                  decoder, context, context_hidden, encoder_optimizer,
                                                  decoder_optimizer, criterion, last)
                encoder_optimizer.step()
                decoder_optimizer.step()
                context_optimizer.step()
                return loss
            else:
                context_hidden = self.trainApair(input_variable, target_variable, encoder,
                                            decoder, context, context_hidden, encoder_optimizer, decoder_optimizer,
                                            criterion, last)

        return 0


    def trainApair(self,input_variable, target_variable,
          encoder, decoder, context, context_hidden,
          encoder_optimizer, decoder_optimizer, criterion,
          last):


        # max_length = self.max_sentence_length
        encoder_hidden = encoder.initHidden()

        # encoder_optimizer.zero_grad() # pytorch accumulates gradients, so zero grad clears them up.
        # decoder_optimizer.zero_grad()

        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]

        encoder_outputs = Variable(torch.zeros(MAX_LENGTH , encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_variable[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = encoder_hidden
        # calculate context
        context_output, context_hidden = context(encoder_output, context_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_output, encoder_outputs, context_hidden)
                # print(len(decoder_output[0]), target_variable[di])
                loss += criterion(decoder_output[0].view(1,-1), target_variable[di].view(1))
                decoder_input = target_variable[di]  # Teacher forcing

        else:

            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_output, encoder_outputs, context_hidden)
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = Variable(torch.LongTensor([[ni]]))
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input

                # only calculate loss if its the last turn
                # print(len(decoder_output[0]), target_variable[di])
                if last:
                    loss += criterion(decoder_output[0].view(1,-1), target_variable[di].view(1))
                if ni == EOS_token:
                    break

        if last:
            loss.backward()

        if last:
            return loss.data[0] / target_length, context_hidden
        else:
            return context_hidden

    def evaluate(self, sentences,max,beam=1):
        max_length = max
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        context_hidden = self.context.initHidden()

        for i, sentence in enumerate(sentences):
            last = False
            if i + 1 == len(sentences):
                last = True
            input_variable = self.dataset.tensorFromSentence(sentence)
            input_length = input_variable.size()[0]
            encoder_hidden = self.encoder.initHidden()

            encoder_outputs = Variable(torch.zeros(max_length, self.encoder.hidden_size))
            encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_variable[ei],
                                                         encoder_hidden)
                encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

            decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            decoder_hidden = encoder_hidden

            # calculate context
            context_output, context_hidden = self.context(encoder_output, context_hidden)

            def decode_with_beam(decoder_inputs, decoder_hiddens, beam):
                new_decoder_inputs = []
                new_decoder_hiddens = []
                decoder_outputs = torch.FloatTensor().cuda() if use_cuda else torch.FloatTensor()
                # decoder_outputs_h = torch.FloatTensor()
                for i, decoder_input in enumerate(decoder_inputs):
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(
                        decoder_input, decoder_hiddens[i], encoder_output, encoder_outputs, context_hidden)
                    # print decoder_output.data
                    # print decoder_outputs
                    decoder_outputs = torch.cat((decoder_outputs, decoder_output.data), 1)
                    # decoder_outputs_h = torch.cat((decoder_outputs_h,decoder_output[0]),1)
                    new_decoder_hiddens.append(decoder_hidden)

                topv, topi = decoder_outputs.topk(beam)
                nis = list(topi[0])
                nh = []  # decoder_hidden
                for ni in nis:
                    nip = ni % len(self.dataset.lang.index2word.keys())  # get the word id
                    # if nip == EOS_token:
                    #    continue # or break?
                    decoder_input = Variable(torch.LongTensor([[nip]]))
                    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
                    new_decoder_inputs.append(decoder_input)
                    nh.append(new_decoder_hiddens[int((ni / len(self.dataset.lang.index2word.keys())))])

                return new_decoder_inputs, nh, (nis[0] % len(self.dataset.lang.index2word.keys())).data

            decoder_inputs = [decoder_input]
            decoder_hiddens = [decoder_hidden]
            for di in range(max_length):
                decoder_inputs, decoder_hiddens, ni = decode_with_beam(decoder_inputs, decoder_hiddens, beam)
                ni = int(ni.data)
                if last:
                    if ni == EOS_token:
                        decoded_words.append('<eos>')
                        break
                    else:
                        decoded_words.append(self.dataset.lang.index2word[ni])

        return decoded_words
