import torch
import madrona_learn
torch.manual_seed(0)

num_channels = 2
batch_size = 2
seq_len = 8

fast_lstm = madrona_learn.rnn.FastLSTM(
    in_channels = num_channels,
    hidden_channels = num_channels * 2,
    num_layers = 1,
)

slow_lstm = madrona_learn.rnn.LSTM(
    in_channels = num_channels,
    hidden_channels = num_channels * 2,
    num_layers = 1,
)

with torch.no_grad():
    for dst, src in zip(slow_lstm.lstm.parameters(), fast_lstm.rnn.parameters()):
        dst.copy_(src)

hidden_start = torch.randn(*slow_lstm.hidden_shape[0:2], batch_size, slow_lstm.hidden_shape[2], dtype=torch.float32)

inputs = torch.randn(seq_len, batch_size, num_channels)

def single_test():
    print("SINGLE TEST\n")
    fast_out, new_fast_hidden = fast_lstm(inputs[0], hidden_start)
    slow_out, new_slow_hidden = slow_lstm(inputs[0], hidden_start)

    print(fast_out)
    print(slow_out)
    print(new_fast_hidden)
    print(new_slow_hidden)

single_test()

print("\n\nSEQUENCE TEST\n")
sequence_breaks = torch.tensor([0, 0, 0, 1, 0, 1, 0, 0], dtype=torch.bool)
print("seq breaks: ", sequence_breaks.to(dtype=torch.uint8))
sequence_breaks = sequence_breaks.view(seq_len, 1, 1).expand(seq_len, batch_size, 1).contiguous()
print(sequence_breaks.shape)

fast_outputs = fast_lstm.fwd_sequence(inputs, hidden_start, sequence_breaks)
print(fast_outputs)

slow_outputs = slow_lstm.fwd_sequence(inputs, hidden_start, sequence_breaks)
print(slow_outputs)

