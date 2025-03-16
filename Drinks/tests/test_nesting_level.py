from django.test import TestCase
from Drinks.complexity_calculator import calculate_nesting_level

class TestNestingLevel(TestCase):

    def test_single_if_statement(self):
        java_code = """
        public class Test {
            public void check(int x) {
                if (x > 10) {
                    System.out.println("Large");
                }
            }
        }
        """
        result = calculate_nesting_level(java_code)
        self.assertEqual(result[3][2], 1)  # Nesting level at if-statement (line 4)

    def test_nested_if_else(self):
        java_code = """
        public class Test {
            public void check(int x) {
                if (x > 10) {
                    if (x < 20) {
                        System.out.println("Medium");
                    }
                } else {
                    System.out.println("Small");
                }
            }
        }
        """
        result = calculate_nesting_level(java_code)
        self.assertEqual(result[3][2], 1)  # Outer if (line 4)
        self.assertEqual(result[4][2], 2)  # Nested if (line 5)

    def test_for_loop(self):
        java_code = """
        public class Test {
            public void loop() {
                for (int i = 0; i < 10; i++) {
                    System.out.println(i);
                }
            }
        }
        """
        result = calculate_nesting_level(java_code)
        self.assertEqual(result[3][2], 1)  # for-loop on line 4

    def test_while_loop(self):
        java_code = """
        public class Test {
            public void loop() {
                int x = 0;
                while (x < 5) {
                    x++;
                }
            }
        }
        """
        result = calculate_nesting_level(java_code)
        self.assertEqual(result[4][2], 1)  # while-loop on line 5

    def test_switch_statement(self):
        java_code = """
        public class Test {
            public void choose(int x) {
                switch (x) {
                    case 1:
                        System.out.println("One");
                        break;
                    case 2:
                        System.out.println("Two");
                        break;
                    default:
                        System.out.println("Other");
                        break;
                }
            }
        }
        """
        result = calculate_nesting_level(java_code)
        self.assertEqual(result[3][2], 1)  # switch (line 4)
        self.assertEqual(result[4][2], 1)  # case 1 (line 5)
        self.assertEqual(result[7][2], 1)  # case 2 (line 8)
        self.assertEqual(result[10][2], 1) # default case (line 11)

    def test_complex_nested_structure(self):
        java_code = """
        public class Test {
            public void complex(int x) {
                if (x > 0) {
                    for (int i = 0; i < x; i++) {
                        while (x < 10) {
                            x++;
                        }
                    }
                }
            }
        }
        """
        result = calculate_nesting_level(java_code)
        self.assertEqual(result[3][2], 1)  # if (line 4)
        self.assertEqual(result[4][2], 2)  # for-loop (line 5)
        self.assertEqual(result[5][2], 3)  # while-loop (line 6)