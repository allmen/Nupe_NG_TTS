import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import yaml

class LSTMEncoder(nn.Module):
    def __init__(self, config_path="config/config.yaml"):
        super().__init__()

        # Load config with safe defaults
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        model_config = config.get('model', {})
        text_config = config.get('text', {})

        self.embedding_dim = model_config.get('embedding_dim', 256)
        self.hidden_dim = model_config.get('encoder_hidden_dim', 256)
        self.dropout = model_config.get('dropout', 0.1)
        self.num_layers = model_config.get('encoder_num_layers', 2)
        self.vocab_size = text_config.get('vocab_size', 100)

        # Character embedding
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embedding_dim,
            padding_idx=0
        )

        # Bi-directional LSTM encoder
        # Use hidden_size = hidden_dim // 2 because of bidirectional concat later
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim // 2,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0
        )

        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, text, text_lengths):
        # text: (batch, seq_len)
        embedded = self.embedding(text)  # (batch, seq_len, embedding_dim)

        # Pack padded sequence (text_lengths must be on CPU)
        packed = pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM
        packed_outputs, (hidden, cell) = self.lstm(packed)

        # Unpack sequence
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        # outputs: (batch, seq_len, hidden_dim) because bidirectional concat

        outputs = self.dropout_layer(outputs)

        # hidden: (num_layers * num_directions, batch, hidden_dim//2)
        # cell: same shape
        return outputs, hidden, cell


class Attention(nn.Module):
    """
    Additive attention that projects encoder outputs and decoder query
    into the same attention space.
    """
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, attention_dim):
        super().__init__()
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.attention_dim = attention_dim

        # Project encoder outputs: (encoder_hidden_dim) -> attention_dim
        self.W_enc = nn.Linear(encoder_hidden_dim, attention_dim, bias=False)
        # Project decoder hidden (query): (decoder_hidden_dim) -> attention_dim
        self.W_dec = nn.Linear(decoder_hidden_dim, attention_dim, bias=False)
        # Scoring vector
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        """
        encoder_outputs: (batch, seq_len, encoder_hidden_dim)
        decoder_hidden: (batch, decoder_hidden_dim)  -- single-step query
        returns:
          context: (batch, encoder_hidden_dim)
          attention_weights: (batch, seq_len)
        """
        # Project encoder outputs -> (batch, seq_len, attention_dim)
        proj_enc = self.W_enc(encoder_outputs)

        # Project decoder hidden -> (batch, attention_dim) then unsqueeze -> (batch, 1, attention_dim)
        proj_dec = self.W_dec(decoder_hidden).unsqueeze(1)

        # Broadcast add and apply tanh then score
        energy = torch.tanh(proj_enc + proj_dec)  # (batch, seq_len, attention_dim)
        scores = self.v(energy).squeeze(2)        # (batch, seq_len)

        attention_weights = F.softmax(scores, dim=1)  # (batch, seq_len)

        # Compute context: weighted sum of encoder outputs
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # (batch, 1, encoder_hidden_dim)
        context = context.squeeze(1)  # (batch, encoder_hidden_dim)

        return context, attention_weights


class Prenet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.layers(x)


class LSTMDecoder(nn.Module):
    def __init__(self, config_path="config/config.yaml"):
        super().__init__()

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        model_config = config.get('model', {})
        audio_config = config.get('audio', {})

        self.hidden_dim = model_config.get('decoder_hidden_dim', 512)
        self.n_mels = audio_config.get('n_mels', 80)
        self.prenet_dim = model_config.get('prenet_dim', 256)
        self.attention_dim = model_config.get('attention_dim', 128)
        self.dropout = model_config.get('dropout', 0.1)
        self.encoder_hidden_dim = model_config.get('encoder_hidden_dim', 256)

        # Prenet: input is previous frame (n_mels) -> prenet_dim
        self.prenet = Prenet(
            self.n_mels,
            self.prenet_dim,
            self.prenet_dim,
            self.dropout
        )

        # Attention: now expects (encoder_hidden_dim, decoder_hidden_dim)
        self.attention = Attention(
            encoder_hidden_dim=self.encoder_hidden_dim,
            decoder_hidden_dim=self.hidden_dim,
            attention_dim=self.attention_dim
        )

        # LSTM decoder (two stacked LSTMCells)
        # Input size to first LSTMCell will be prenet_dim + encoder_hidden_dim (context)
        self.lstm1 = nn.LSTMCell(
            self.prenet_dim + self.encoder_hidden_dim,
            self.hidden_dim
        )
        self.lstm2 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

        # Project encoder final hidden (concatenated forward/backward) -> decoder hidden size
        # NOTE: encoder_final_hidden already has size encoder_hidden_dim (not *2)
        self.encoder_projection = nn.Linear(
            self.encoder_hidden_dim,
            self.hidden_dim
        )

        self.dropout_layer = nn.Dropout(self.dropout)

        # Output projections: input is decoder hidden (hidden_dim) + context (encoder_hidden_dim)
        self.mel_projection = nn.Linear(
            self.hidden_dim + self.encoder_hidden_dim,
            self.n_mels
        )
        self.stop_projection = nn.Linear(
            self.hidden_dim + self.encoder_hidden_dim,
            1
        )

    def forward(self, encoder_outputs, encoder_hidden, targets=None,
                teacher_forcing_ratio=1.0):
        """
        encoder_outputs: (batch, seq_len, encoder_hidden_dim)
        encoder_hidden: tuple(hidden, cell) where hidden shape is
                        (num_layers * num_directions, batch, encoder_hidden_dim//2)
        """
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device

        hidden_3d, cell_3d = encoder_hidden  # both: (num_layers * num_directions, batch, hidden_per_dir)
        # Reconstruct encoder final hidden by taking last layer and concatenating directions
        # Expecting num_layers and num_directions same as encoder
        # We attempt to infer num_layers and num_directions
        total_layers = hidden_3d.size(0)
        # If bidirectional and num_layers known as 2, common layout: (num_layers * 2, batch, hidden_per_dir)
        # Try to find num_directions by checking hidden_3d dims: if total_layers is even assume 2 directions
        num_directions = 2 if total_layers % 2 == 0 else 1
        num_layers = total_layers // num_directions
        hidden_per_dir = hidden_3d.size(2)

        # Reshape: (num_layers, num_directions, batch, hidden_per_dir)
        hidden_reshaped = hidden_3d.view(num_layers, num_directions, batch_size, hidden_per_dir)
        # Take last layer
        last_layer_hidden = hidden_reshaped[-1]  # (num_directions, batch, hidden_per_dir)
        # Permute to (batch, num_directions, hidden_per_dir)
        last_layer_hidden = last_layer_hidden.permute(1, 0, 2)
        # Concatenate directions -> (batch, encoder_hidden_dim)
        encoder_final_hidden = last_layer_hidden.reshape(batch_size, self.encoder_hidden_dim)

        # Project encoder final hidden to initialize decoder hidden state
        projected_hidden = self.encoder_projection(encoder_final_hidden)  # (batch, hidden_dim)

        # Initialize decoder states
        hidden1 = projected_hidden
        cell1 = torch.zeros(batch_size, self.hidden_dim, device=device)
        hidden2 = torch.zeros(batch_size, self.hidden_dim, device=device)
        cell2 = torch.zeros(batch_size, self.hidden_dim, device=device)

        # First input frame (all zeros)
        decoder_input = torch.zeros(batch_size, self.n_mels, device=device)

        mel_outputs = []
        stop_outputs = []
        alignments = []

        max_length = 300 if targets is None else targets.size(1)

        for t in range(max_length):
            # Prenet from previous frame
            prenet_out = self.prenet(decoder_input)  # (batch, prenet_dim)

            # Attention: use decoder query = hidden1 + hidden2 (both decoder hidden_dim)
            attention_query = hidden1 + hidden2  # (batch, hidden_dim)
            # Attention expects decoder_hidden of decoder_hidden_dim; it internally projects to attention space
            context, attention_weights = self.attention(encoder_outputs, attention_query)
            alignments.append(attention_weights)

            # LSTMCell step
            lstm1_input = torch.cat([prenet_out, context], dim=1)  # (batch, prenet_dim + encoder_hidden_dim)
            hidden1, cell1 = self.lstm1(lstm1_input, (hidden1, cell1))
            hidden1 = self.dropout_layer(hidden1)

            hidden2, cell2 = self.lstm2(hidden1, (hidden2, cell2))
            hidden2 = self.dropout_layer(hidden2)

            projection_input = torch.cat([hidden2, context], dim=1)  # (batch, hidden_dim + encoder_hidden_dim)
            mel_output = self.mel_projection(projection_input)
            stop_output = torch.sigmoid(self.stop_projection(projection_input))

            mel_outputs.append(mel_output)
            stop_outputs.append(stop_output)

            # Teacher forcing: feed ground-truth mel if available with probability teacher_forcing_ratio
            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = targets[:, t, :].to(device)
            else:
                decoder_input = mel_output

        # Stack lists -> tensors
        mel_outputs = torch.stack(mel_outputs, dim=1)   # (batch, seq_len, n_mels)
        stop_outputs = torch.stack(stop_outputs, dim=1) # (batch, seq_len, 1)
        alignments = torch.stack(alignments, dim=1)    # (batch, seq_len, encoder_seq_len)

        return mel_outputs, stop_outputs, alignments


class LSTMTTS(nn.Module):
    def __init__(self, config_path="config/config.yaml"):
        super().__init__()

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.encoder = LSTMEncoder(config_path)
        self.decoder = LSTMDecoder(config_path)

    def forward(self, text, text_lengths, mel_targets=None):
        # Encode text
        encoder_outputs, hidden, cell = self.encoder(text, text_lengths)

        # Decode to mel-spectrogram
        teacher_forcing_ratio = self.config.get('model', {}).get('teacher_forcing_ratio', 1.0)
        if self.training:
            mel_outputs, stop_outputs, alignments = self.decoder(
                encoder_outputs, (hidden, cell), mel_targets, teacher_forcing_ratio
            )
        else:
            mel_outputs, stop_outputs, alignments = self.decoder(
                encoder_outputs, (hidden, cell), None, 0.0
            )

        return mel_outputs, stop_outputs, alignments

    def inference(self, text, text_lengths):
        with torch.no_grad():
            mel_outputs, stop_outputs, alignments = self.forward(
                text, text_lengths, None
            )
        return mel_outputs, stop_outputs, alignments
