import unittest

from training.build_data import Cleaner, normalize_quotes
from training.common import MAX_TOKENS, restore_from_source

# T5's view of a token: characters outside its vocabulary vanish, NFKC changes others
LOST = set('ã`<{}~^') | {'ñ'}


def t5_view(token):
    return ''.join(c for c in token.replace('…', '...') if c not in LOST)


class FakeTokenizer:
    def __call__(self, text):
        return {'input_ids': text.split()}


class RestoreFromSource(unittest.TestCase):
    def test_restores_lost_characters_in_copied_tokens(self):
        self.assertEqual(restore_from_source('I live in São Paulo .', 'I live in So Paulo .', t5_view),
                         'I live in São Paulo .')

    def test_keeps_corrections(self):
        self.assertEqual(restore_from_source('He go to São Paulo .', 'He goes to So Paulo .', t5_view),
                         'He goes to São Paulo .')

    def test_reinserts_tokens_that_vanish_entirely(self):
        self.assertEqual(restore_from_source('if x < y then', 'if x y then', t5_view), 'if x < y then')
        self.assertEqual(restore_from_source('< hello', 'hello', t5_view), '< hello')

    def test_normalized_characters(self):
        self.assertEqual(restore_from_source('family … were', 'family ... was', t5_view), 'family … was')

    def test_source_without_lossy_tokens_is_left_alone(self):
        self.assertEqual(restore_from_source('He go home .', 'He goes  home .', t5_view), 'He goes home .')


class Cleaning(unittest.TestCase):
    def test_normalize_quotes(self):
        self.assertEqual(normalize_quotes("I watched `` Grey 's anatomy '' today ."),
                         'I watched " Grey \'s anatomy " today .')

    def test_default_filters(self):
        cleaner = Cleaner(FakeTokenizer())
        long = ' '.join(['w'] * (MAX_TOKENS + 1))
        pairs = [('a b', 'a c', 'x'), ('a b', '', 'x'), (long, long, 'x'), ('a b', 'a c', 'x'),
                 ('one', 'one two three four five', 'x')]
        kept = cleaner.clean(pairs)
        # duplicates and length-ratio outliers are kept unless asked for
        self.assertEqual(kept, [('a b', 'a c', 'x'), ('a b', 'a c', 'x'), ('one', 'one two three four five', 'x')])
        self.assertEqual(dict(cleaner.dropped), {'empty': 1, f'over_{MAX_TOKENS}_tokens': 1})

    def test_opt_in_filters(self):
        cleaner = Cleaner(FakeTokenizer(), ratio_filter=True, dedupe=True)
        kept = cleaner.clean([('a b', 'a c', 'x'), ('a b', 'a c', 'x'), ('one', 'one two three four five', 'x')])
        self.assertEqual(kept, [('a b', 'a c', 'x')])
        self.assertEqual(dict(cleaner.dropped), {'duplicate': 1, 'length_ratio': 1})


if __name__ == '__main__':
    unittest.main()
