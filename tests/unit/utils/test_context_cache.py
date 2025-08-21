import unittest

from dnois.utils import ContextCache, context_cache, enable_group_cache


class ContextCacheSub(ContextCache):
    def __init__(self):
        self.called = []

    @context_cache
    def foo(self):
        self.called.append('foo')
        return 'foo'

    @context_cache('BAR')
    def bar(self):
        self.called.append('bar')
        return 'bar'

    def baz(self):
        self.called.append('baz')
        return 'baz'


class TestContextCache(unittest.TestCase):
    def test_context_cache(self):
        sub = ContextCacheSub()
        with sub.enable_cache(['foo', 'bar']):
            self.assertEqual(sub.foo(), 'foo')
            self.assertListEqual(sub.called, ['foo'])

            self.assertEqual(sub.foo(), 'foo')
            self.assertListEqual(sub.called, ['foo'])
            sub.called.clear()

            self.assertEqual(sub.bar(), 'bar')
            self.assertListEqual(sub.called, ['bar'])
            self.assertEqual(sub.bar(), 'bar')
            self.assertListEqual(sub.called, ['bar', 'bar'])
            sub.called.clear()

            sub.baz()
            sub.baz()
            self.assertListEqual(sub.called, ['baz', 'baz'])
            sub.called.clear()

    def test_cache_cleared_after_context(self):
        sub = ContextCacheSub()
        with sub.enable_cache(['foo']):
            sub.foo()
            self.assertListEqual(sub.called, ['foo'])

        sub.foo()
        self.assertListEqual(sub.called, ['foo', 'foo'])

    def test_enable_group_cache(self):
        from dnois.utils import enable_group_cache
        
        # Create multiple cache instances
        cache1 = ContextCacheSub()
        cache2 = ContextCacheSub()
        
        # Test group cache functionality
        with enable_group_cache(['foo', 'bar'], [cache1, cache2]) as entries:
            # First calls should execute and cache
            result1 = cache1.foo()
            result2 = cache2.bar()
            self.assertEqual(result1, 'foo')
            self.assertEqual(result2, 'bar')
            self.assertEqual(cache1.called, ['foo'])
            self.assertEqual(cache2.called, ['bar'])
            
            # Second calls should use cache
            result3 = cache1.foo()
            result4 = cache2.bar()
            self.assertEqual(result3, 'foo')
            self.assertEqual(result4, 'bar')
            self.assertEqual(cache1.called, ['foo'])  # No new call
            self.assertEqual(cache2.called, ['bar', 'bar'])  # new call
            
            # Check that entries are returned
            self.assertEqual(len(entries), 2)
            self.assertIsInstance(entries[0], dict)
            self.assertIsInstance(entries[1], dict)
