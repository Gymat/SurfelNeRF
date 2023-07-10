import math
import typing
import torch


class DictArray:
    def __init__(self, dicts: typing.List[dict], types={}, *, batch_size: int = 1024,
                 string_encoding: typing.Literal['ascii', 'utf_16_le', 'utf_32_le'] = 'utf_16_le',
                 ints_dtype=torch.int64):
        dicts = list(dicts)
        numel = len(dicts)
        assert numel > 0
        self.tensors = {k: t(numel) for k, t in types.items() if t != StringArray and t != IntsArray}
        string_lists = {k: [None] * numel for k, t in types.items() if t == StringArray}
        ints_lists = {k: [None] * numel for k, t in types.items() if t == IntsArray}
        temp_lists = {k: [None] * batch_size for k in self.tensors}
        for b in range(math.ceil(numel / batch_size)):
            for i, t in enumerate(dicts[b * batch_size: (b + 1) * batch_size]):
                for k in temp_lists:
                    temp_lists[k][i] = t[k]
                for k in string_lists:
                    string_lists[k][b * batch_size + i] = t[k]
                for k in ints_lists:
                    ints_lists[k][b * batch_size + i] = t[k]
            for k, v in temp_lists.items():
                res = self.tensors[k][b * batch_size: (b + 1) * batch_size]
                res.copy_(torch.as_tensor(v[:len(res)], dtype=self.tensors[k].dtype))
        self.string_arrays = {k: StringArray(v, encoding=string_encoding) for k, v in string_lists.items()}
        self.ints_arrays = {k: IntsArray(v, dtype=ints_dtype) for k, v in ints_lists.items()}

    def __getitem__(self, i):
        return dict(**{k: v[i].item() for k, v in self.tensors.items()},
                    **{k: v[i] for k, v in self.string_arrays.items()},
                    **{k: v[i] for k, v in self.ints_arrays.items()})

    def __len__(self):
        return len(next(iter(self.tensors.values()))) if len(self.tensors) > 0 else len(
            next(iter(self.string_arrays.values())))


class NamedTupleArray(DictArray):
    def __init__(self, namedtuples, *args, **kwargs):
        super().__init__([t._asdict() for t in namedtuples], *args, **kwargs)
        self.namedtuple = type(next(iter(namedtuples)))

    def __getitem__(self, index):
        return self.namedtuple(**super().__getitem__(index))


class StringArray:
    def __init__(self, strings: typing.List[str],
                 encoding: typing.Literal['ascii', 'utf_16_le', 'utf_32_le'] = 'utf_16_le'):
        strings = list(strings)
        self.encoding = encoding
        self.multiplier = dict(ascii=1, utf_16_le=2, utf_32_le=4)[encoding]
        self.data = torch.ByteTensor(torch.ByteStorage.from_buffer(''.join(strings).encode(encoding)))
        self.cumlen = torch.LongTensor(list(map(len, strings))).cumsum(dim=0).mul_(self.multiplier)
        assert int(self.cumlen[-1]) == len(
            self.data), f'[{encoding}] is not enough to hold characters, use a larger character class'

    def __getitem__(self, i):
        return bytes(self.data[(self.cumlen[i - 1] if i >= 1 else 0): self.cumlen[i]]).decode(self.encoding)

    def __len__(self):
        return len(self.cumlen)

    def tolist(self):
        data_bytes, cumlen = bytes(self.data), self.cumlen.tolist()
        return [data_bytes[0:cumlen[0]].decode(self.encoding)] + [data_bytes[start:end].decode(self.encoding) for
                                                                  start, end in zip(cumlen[:-1], cumlen_mul[1:])]


class IntsArray:
    def __init__(self, ints, dtype=torch.int64):
        tensors = [torch.as_tensor(t, dtype=dtype) for t in ints]
        self.data = torch.cat(tensors)
        self.cumlen = torch.tensor(list(map(len, tensors)), dtype=torch.int64).cumsum(dim=0)

    def __getitem__(self, i):
        return self.data[(self.cumlen[i - 1] if i >= 1 else 0): self.cumlen[i]]

    def __len__(self):
        return len(self.cumlen)


if __name__ == '__main__':
    a = StringArray(['asd', 'def'])
    print('len = ', len(a))
    print('data = ', list(a))

    a = DictArray([dict(a=1, b='def'), dict(a=2, b='klm')], types=dict(a=torch.LongTensor, b=StringArray))
    print('len = ', len(a))
    print('data = ', list(a))