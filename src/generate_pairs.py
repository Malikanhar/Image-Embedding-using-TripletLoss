#! encoding: utf-8

import os
import random
import argparse

class GeneratePairs:
    """
    Generate the pairs.txt file that is used for training face classifier when calling python `src/train_softmax.py`.
    Or others' python scripts that needs the file of pairs.txt.

    Doc Reference: http://vis-www.cs.umass.edu/lfw/README.txt
    """

    def __init__(self, data_dir, pairs_filepath):
        """
        Parameter data_dir, is your data directory.
        Parameter pairs_filepath, where is the pairs.txt that belongs to.
        """
        self.data_dir = data_dir
        self.pairs_filepath = pairs_filepath

    def generate(self):
        matches_pairs = self._generate_matches_pairs()
        mismatches_pairs = self._generate_mismatches_pairs(len(matches_pairs))
        pairs_in_1_fold = int(len(matches_pairs) / 10)
        all_pairs = []
        for i in range(10):
            all_pairs += \
                matches_pairs[i*pairs_in_1_fold:(i+1)*pairs_in_1_fold] + \
                mismatches_pairs[i*pairs_in_1_fold:(i+1)*pairs_in_1_fold]
        print("Matches pairs: " + str(len(matches_pairs)))
        print("Mismatches pairs: " + str(len(mismatches_pairs)))
        print("Total pairs: " + str(len(all_pairs)))
        with open(self.pairs_filepath, "w") as f:
            f.write("10\t" + str(len(all_pairs)) + "\n")
            for pair in all_pairs:
                f.write(pair)

    def _generate_matches_pairs(self):
        """
        Generate all matches pairs
        """
        dir_list = os.listdir(self.data_dir)
        dir_list.sort()
        pairs_text = []
        for name in dir_list:
            if name in (".DS_Store", "pairs.txt"):
                continue

            files_count = []
            for file in os.listdir(self.data_dir + name):
                if file == ".DS_Store":
                    continue
                files_count.append(file)
            if len(files_count) <= 1:
                continue
            elif len(files_count) == 2:
                pairs_count = 1
            elif len(files_count) == 3:
                pairs_count = 3
            else:
                pairs_count = 5

            pairs = []
            while len(pairs) < pairs_count:
                # This line may vary depending on how your images are named.
                temp = random.choice(files_count).split("_")
                w = "_".join(temp[:-1])
                l = os.path.splitext(random.choice(files_count).split(
                    "_")[-1].lstrip("0"))[0]
                r = os.path.splitext(random.choice(files_count).split(
                    "_")[-1].lstrip("0"))[0]
                while r == l:
                    r = os.path.splitext(random.choice(files_count).split(
                        "_")[-1].lstrip("0"))[0]
                if l + r in pairs or r + l in pairs:
                    continue
                pairs_text.append(w + "\t" + l + "\t" + r + "\n")
                pairs.append(l + r)
        while len(pairs_text) % 10:
            pairs_text.pop(random.randrange(0, len(pairs_text)))
        return pairs_text

    def _generate_mismatches_pairs(self, matches_pairs_count: int):
        """
        Generate all mismatches pairs
        """
        dir_list = os.listdir(self.data_dir)
        dir_list = [f_n for f_n in dir_list if f_n not in (
            ".DS_Store", "pairs.txt")]
        dir_list.sort()
        all_name_pairs = []
        pairs_text = []
        for name in dir_list:
            other_name = random.choice(dir_list)
            while (
                    other_name == name or
                    name + other_name in all_name_pairs or
                    other_name + name in all_name_pairs):
                other_name = random.choice(dir_list)
            files_1 = os.listdir(self.data_dir + name)
            files_2 = os.listdir(self.data_dir + other_name)
            pair_files_count = len(files_1) + len(files_2)
            if pair_files_count <= 1:
                continue
            elif pair_files_count >= 6:
                pairs_count = 5
            else:
                pairs_count = pair_files_count - 1
            pairs = []
            while len(pairs) < pairs_count:
                file_1 = random.choice(files_1)
                wl = "_".join(file_1.split("_")[:-1])
                l = os.path.splitext(file_1.split(
                    "_")[-1].lstrip("0"))[0]
                file_2 = random.choice(files_2)
                wr = "_".join(file_2.split("_")[:-1])
                r = os.path.splitext(file_2.split(
                    "_")[-1].lstrip("0"))[0]
                if l + r in pairs or r + l in pairs:
                    continue
                pairs_text.append(wl + "\t" + l + "\t" + wr + "\t" + r + "\n")
                pairs.append(l + r)
            all_name_pairs.append(name + other_name)
        assert len(
            pairs_text) >= matches_pairs_count, "len(pairs_text) < matches_pairs_count"
        while len(pairs_text) > matches_pairs_count:
            pairs_text.pop(random.randrange(0, len(pairs_text)))
        return pairs_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser to Generate Pairs')
    parser.add_argument('--data_dir', type=str, required=True,
                                    help='Path to ordered image dataset')
    parser.add_argument('--pairs', type=str, default='pairs.txt',
                                    help='Output txt filename')
    args = parser.parse_args()
    data_dir = args.data_dir
    generatePairs = GeneratePairs(args.data_dir, args.pairs)
    generatePairs.generate()
