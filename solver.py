import numpy as np
import pandas as pd
import json

with open('words.json', 'r') as f:
    all_words: dict = json.load(f)


def get_crossing(i, j, p, q):
    return ((i + p) // 2, (j + q) // 2)


def get_available_neighbours(i, j, visited, crossed):
        neighbours = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                position = (i + di, j + dj)
                if position[0] < 0 or position[0] >= visited.shape[0]:
                    continue
                if position[1] < 0 or position[1] >= visited.shape[1]:
                    continue
                if visited[position]:
                    continue            

                if abs(di) == 1 and abs(dj) == 1:
                    crossing = get_crossing(i, j, *position)
                    if crossed[crossing]:
                        continue
                
                neighbours.append(position)
        return neighbours


def get_mean_log_frequency(words):
    return np.mean([all_words[word.lower()] for word in words])


class Solver:

    def __init__(self, board, n_words, 
                 n_words_tolerance = 2,
                 min_word_length = 5,
                 min_log_freq = 4.8,
                 known_words = [], 
                 forbidden_words = [], 
                 ratio_tolerance = 0.937, # 45/48 3 letters may be missing
                 unique_word_tolerance = 1.5,
                 stop_on_first_full_coverage = True,
                 verbose = 0,
                 ):
        self.board = board

        self.n_words = n_words
        self.n_words_tolerance = n_words_tolerance
        self.min_words = n_words - n_words_tolerance
        self.max_words = n_words + n_words_tolerance

        self.min_word_length = min_word_length
        self.min_log_freq = min_log_freq

        self.ratio_tolerance = ratio_tolerance
        self.unique_word_tolerance = unique_word_tolerance
        self.known_words = known_words
        self.forbidden_words = set(forbidden_words)
        self.stop_on_first_full_coverage = stop_on_first_full_coverage
        self.verbose = verbose

        self.max_length = 0

        self.__get_possible_words()
        self.__find_possible_words()

        coverage_params = self.__get_initial_coverage()
        self.solutions = []
        coverage_params = self.__sort_words(*coverage_params)
        self.aborted = False
        self.__cover_board(*coverage_params)
        if self.verbose > 0:
            print(f"found {len(self.solutions)} solutions, cleaning up...")
        self.__clean_solutions()
        if self.verbose > 0:
            print(f"found {len(self.solutions)} viable solutions...")
        self.__save_solutions()


    def __get_possible_words(self):
        self.words = {}
        self.prefixes = set()
        board_set = set(self.board.flatten())
        for word, log_freq in all_words.items():
            if word.upper() in self.forbidden_words:
                continue
            if len(word) < self.min_word_length:
                continue
            if log_freq < self.min_log_freq:
                continue
            if set(word.upper()) <= board_set:
                self.max_length = max(self.max_length, len(word))
                self.words[word.upper()] = log_freq
                for p in range(len(word)):
                    self.prefixes.add(word[:p+1].upper())
        if self.verbose > 0:
            print(f"found {len(self.words)} possible words")


    def __find_possible_words(self):
        self.found_words = []
        self.masks = []
        self.diagonals = []
        self.starts = []
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if self.verbose > 1:
                    print(f"checking position {i + 1}, {j + 1}...")
                visted = np.zeros_like(self.board, dtype=np.bool_)
                crossed = np.zeros((self.board.shape[0] - 1, self.board.shape[1] - 1), dtype=np.bool_)
                self.__explore(i, j, "", visted, crossed, (i, j))
        self.found_words = np.array(self.found_words)
        if self.verbose > 0:
            print(f"found {len(self.found_words)} words on the board")

        # clean up duplicates
        value, counts = np.unique(self.found_words, return_counts=True)
        element_counts = dict(zip(value, counts))
        unique = [key for key, value in element_counts.items() if value < 2]
        non_unique = [key for key, value in element_counts.items() if value >= 2]
        non_unique_idx = np.where(np.isin(self.found_words, non_unique))[0]

        checked = set()
        for idx in non_unique_idx:
            non_unique_word = self.found_words[idx]
            if non_unique_word in checked:
                continue
            checked.add(non_unique_word)
            global_mask = np.any(np.stack(self.masks)[np.where(self.found_words == non_unique_word)], axis=0)
            if (np.sum(global_mask) <= self.unique_word_tolerance * len(non_unique_word)) \
                    or non_unique_word in self.known_words:
                unique.append(non_unique_word)

        unique_idx = np.where(np.isin(self.found_words, unique))[0]

        self.found_words = self.found_words[unique_idx]
        self.masks = np.stack(self.masks)[unique_idx]
        self.diagonals = np.stack(self.diagonals)[unique_idx]
        self.starts = np.array(self.starts)[unique_idx]
        if self.verbose > 0:
            print(f"found {len(self.found_words)} sufficently unique words on the board")


    def __explore(self, i, j, string, visited, crossed, start):
        string += self.board[i, j]
        visited[i, j] = True
        if string not in self.prefixes:
            return
        if string in self.words:
            self.found_words.append(string)
            self.masks.append(visited * 1)
            self.diagonals.append(crossed * 1)
            self.starts.append(start)
        if len(string) >= self.max_length:
            return
        
        neighbours = get_available_neighbours(i, j, visited, crossed)
        for neighbour in neighbours:
            is_diagonal = abs(neighbour[0] - i) == 1 and abs(neighbour[1] - j) == 1
            crossing = get_crossing(i, j, *neighbour) if is_diagonal else None
            if crossing is not None:
                crossed[crossing] = True
            self.__explore(neighbour[0], neighbour[1], string, np.copy(visited), np.copy(crossed), start)
            if crossing is not None:
                crossed[crossing] = False


    def __get_initial_coverage(self):
        coverage = np.zeros_like(self.masks[0])
        diagonal_coverage = np.zeros_like(self.diagonals[0])
        used_words = []
        for word in self.known_words:
            # if np.count_nonzero(self.found_words == word) > 1:
            #     print(f"known word ({word}) is not unique, omitting from initial coverage...")
            #     continue
            idx = np.argmax(self.found_words == word)
            word_mask = self.masks[idx, :, :]
            word_diagonals = self.diagonals[idx, :, :]
            tmp_diagonal_coverage = diagonal_coverage + word_diagonals
            if np.any(tmp_diagonal_coverage > 1):
                print(f"known word ({word}) violates diagonals, omitting from initial coverage...")
                continue
            diagonal_coverage = tmp_diagonal_coverage
            coverage += word_mask
            used_words.append(word)
        legal_idx_masks = np.where(np.all(self.masks * coverage == 0, axis=(1, 2)))[0]
        legal_idx_diagonals = np.where(np.all(self.diagonals * diagonal_coverage == 0, axis=(1, 2)))[0]
        legal_idx = np.intersect1d(legal_idx_masks, legal_idx_diagonals)
        return used_words, coverage, diagonal_coverage, \
            self.found_words[legal_idx], self.masks[legal_idx], self.diagonals[legal_idx], self.starts[legal_idx]


    def __sort_words(self, used_words, coverage, diagonal_coverage, words, masks, diagonals, starts):
        def potential_spangram(word_mask):
            column_coverage = np.max(word_mask, axis=0)
            if column_coverage[0] == 1 and column_coverage[-1] == 1:
                return -2_000_000
            row_coverage = np.max(word_mask, axis=1)
            if row_coverage[0] == 1 and row_coverage[-1] == 1:
                return -2_000_000
            return 0
        
        def starts_on_wall(word_start):
            score = 0
            i, j = word_start
            if i == 0 or i == self.board.shape[0] - 1:
                score += -500_000
            if j == 0 or j == self.board.shape[1] - 1:
                score += -500_000
            return score
        
        def touches_corner(word_mask):
            if word_mask[0, 0] == 1:
                return -1_000_000
            if word_mask[0, -1] == 1:
                return -1_000_000
            if word_mask[-1, 0] == 1:
                return -1_000_000
            if word_mask[-1, -1] == 1:
                return -1_000_000
            return 0
            
        def elimination_estimate(word, word_mask, word_diagonals, depth = 0):
            legal_idx_masks = np.where(np.all(masks * word_mask == 0, axis=(1, 2)))[0]
            legal_idx_diagonals = np.where(np.all(diagonals * word_diagonals == 0, axis=(1, 2)))[0]
            legal_idx = np.intersect1d(legal_idx_masks, legal_idx_diagonals)

            return len(legal_idx) / len(words)


        def get_value(word, word_mask, word_diagonals, word_start):
            score = 0
            score += potential_spangram(word_mask)
            # score += starts_on_wall(word_start)
            score += touches_corner(word_mask)
            score += elimination_estimate(word, word_mask, word_diagonals)
            return score
        
        values = []
        for i in range(len(words)):
            value = get_value(words[i], masks[i], diagonals[i], starts[i])
            values.append(value)
            if self.verbose > 2:
                print(f"({i + 1}/{len(words)}) {words[i]}: {value}")
        idx = np.argsort(values)

        words = words[idx]
        masks = masks[idx]
        diagonals = diagonals[idx]
        starts = starts[idx]

        return used_words, coverage, diagonal_coverage, words, masks, diagonals, starts


    def __cover_board(self, used_words, coverage, diagonal_coverage, words, masks, diagonals, starts, depth=0):
        if self.verbose > 3:
            indent = '\t' * depth
            print(f"{indent} depth {depth} ({used_words[-1] if len(used_words) > 0 else ''}), {len(words)} possible words")
        original_words = np.copy(words)
        original_masks = np.copy(masks)
        original_diagonals = np.copy(diagonals)
        original_starts = np.copy(starts)
        for i, word in enumerate(original_words):
            if self.aborted:
                return
            if depth == 0 and self.verbose > 1:
                print(f"trying word {i + 1}/{len(original_words)} ({word}, {all_words[word.lower()]:.2f}) as starting word... \
                      ({len(self.solutions)} solutions found so far)")
            # ignore all words before
            words = original_words[i:]
            masks = original_masks[i:]
            diagonals = original_diagonals[i:]
            starts = original_starts[i:]

            word_mask = original_masks[i, :, :]
            word_diagonals = original_diagonals[i, :, :]
            
            used_words.append(word)
            coverage += word_mask
            diagonal_coverage += word_diagonals

            legal_idx_masks = np.where(np.all(masks * coverage == 0, axis=(1, 2)))[0]
            legal_idx_diagonals = np.where(np.all(diagonals * diagonal_coverage == 0, axis=(1, 2)))[0]
            legal_idx = np.intersect1d(legal_idx_masks, legal_idx_diagonals)

            potential_coverage = np.where(np.sum(masks[legal_idx], axis=0) + coverage > 0, 1, 0)
            leads_to_gaps = np.mean(potential_coverage) < self.ratio_tolerance

            if len(legal_idx) > 0 \
                    and not leads_to_gaps \
                    and len(used_words) < self.max_words\
                    and len(legal_idx) + len(used_words) >= self.min_words:
                self.__cover_board(
                    used_words, coverage, diagonal_coverage,
                    words[legal_idx], masks[legal_idx], diagonals[legal_idx], starts[legal_idx],
                    depth+1)
            else:
                ratio = np.mean(coverage)
                if self.stop_on_first_full_coverage and ratio == 1.0:
                    if self.verbose > 0:
                        print("A full coverage was found!")
                    self.solutions = [(np.copy(used_words), ratio)]
                    self.__save_solutions(top_n=1, include_log_freq=False)
                    self.aborted = True
                    return
                self.solutions.append((np.copy(used_words), ratio))
            
            # reset
            diagonal_coverage -= word_diagonals
            coverage -= word_mask
            used_words.pop()


    def __clean_solutions(self):
        filtered_solutions = []
        for words, ratio in self.solutions:
            if ratio < self.ratio_tolerance:
                continue
            if len(words) < self.min_words \
                or len(words) > self.max_words:
                continue
            filtered_solutions.append((words, ratio, get_mean_log_frequency(words)))

        self.solutions = sorted(filtered_solutions, key=lambda s : (-s[1], -s[2]))


    def __save_solutions(self, top_n = 10, include_log_freq = True):
        with open('./output.txt', 'w') as f:
            for solution in self.solutions[:top_n]:
                for word in solution[0]:
                    f.write(f"{word}, ")
                if include_log_freq:
                    f.write(f"{solution[1]:.3f}, {solution[2]:.3f}\n")
                else:
                    f.write(f"{solution[1]:.3f}\n")


def run():
    board = [
        "O	R	D	B	A	O",
        "A	F	E	K	T	E",
        "E	S	O	A	D	H",
        "D	T	N	T	S	D",
        "T	D	O	A	M	E",
        "N	A	P	T	S	P",
        "D	E	O	C	L	O",
        "I	R	F	H	A	L",
    ]

    board = np.array([line.split('\t') for line in board])
    solver = Solver(board, 
                    n_words=7,
                    n_words_tolerance=1,
                    min_word_length=3,
                    min_log_freq=5,
                    verbose=2, 
                    known_words=[],
                    forbidden_words=[], 
                    ratio_tolerance=1.0,
                    unique_word_tolerance=1.8,
                    stop_on_first_full_coverage=True)
    
def main():
    run()


if __name__ == '__main__':
    main()