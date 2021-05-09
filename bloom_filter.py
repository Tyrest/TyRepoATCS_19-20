import random

class HashFamily:
    def __init__(self):
        self.memomask = {}
    def hash_fn(self, n):
        mask = self.memomask.get(n)
        if mask is None:
            random.seed(n)
            mask = self.memomask[n] = random.getrandbits(64)
        return lambda x: hash(x) ^ mask

class BloomFilter:
    def __init__(self, num_hashes, num_slots):
        hf = HashFamily()
        self.hash_fn_list = [hf.hash_fn(n) for n in range(num_hashes)]
        self.num_slots = num_slots
        self.vector = [0]*num_slots

    # Implement this method.
    # It should apply all of the hash functions to the string s
    # and then for each result value of the hash function (modulo num_slots),
    # set the value at the matching index in the vector to 1.
    def AddString(self, s):
        for fn in self.hash_fn_list:
            self.vector[fn(s) % self.num_slots] = 1
        pass

    # Implement this method.
    # It should return the string "Not Member" or "Maybe Member", as appropriate.
    def IsMember(self, q):
        for fn in self.hash_fn_list:
            if self.vector[fn(q) % self.num_slots] != 1:
                return "Not Member"
        return "Maybe Member"

def main():
    num_hashes = 4
    num_slots = 128
    bf = BloomFilter(num_hashes, num_slots)
    initial_set = ["potato", "tomato", "hippopotamus", "rhinoceros"]
    for word in initial_set:
        bf.AddString(word)

    assert(bf.IsMember("turnip") == "Not Member")
    assert(bf.IsMember("potato") == "Maybe Member")

if __name__ == '__main__':
    main()