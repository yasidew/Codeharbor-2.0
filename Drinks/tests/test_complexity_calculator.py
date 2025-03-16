from django.test import TestCase
from Drinks.complexity_calculator import calculate_control_structure_complexity

class ComplexityCalculatorTestCase(TestCase):
    def test_if_statement(self):
        java_code = """
        public class Test {
            public void check(int x) {
                if (x > 10) {
                    System.out.println("Large");
                }
            }
        }
        """
        line_weights, total_weight = calculate_control_structure_complexity(java_code)
        self.assertEqual(total_weight, 1)
        self.assertIn(4, line_weights)

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
        line_weights, total_weight = calculate_control_structure_complexity(java_code)
        self.assertEqual(total_weight, 2)
        self.assertIn(4, line_weights)

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
        line_weights, total_weight = calculate_control_structure_complexity(java_code)
        self.assertEqual(total_weight, 3)
        self.assertIn(4, line_weights)