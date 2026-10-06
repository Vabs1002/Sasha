#!/usr/bin/env python3
"""
Basic functionality test for Sasha AI Interviewer
"""
import os
import sys
import tempfile

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that core modules can be imported"""
    try:
        import resume_parser
        import signal_detector
        import analyzer
        import interviewer_agent
        print("[PASS] Core modules imported successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False

def test_resume_parser():
    """Test resume parser with sample text file"""
    try:
        from resume_parser import parse_resume

        # Create a temporary text file to test with
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""
            John Doe
            Software Engineer

            Experience:
            January 2020 - Present: Software Engineer at Tech Corp
              - Built web applications using Python and React
              - Improved system performance by 30%

            Skills: Python, JavaScript, React

            Projects:
            - Built a recommendation engine using machine learning
            """)
            temp_file = f.name

        try:
            # This will fail because we're trying to parse .txt as .pdf/.docx
            # but we want to test that it handles unsupported formats gracefully
            result = parse_resume(temp_file)
            print("[FAIL] Should have failed for unsupported format")
            return False
        except ValueError as e:
            if "Unsupported file type" in str(e):
                print("[PASS] Resume parser correctly rejects unsupported formats")
                return True
            else:
                print(f"[FAIL] Unexpected error: {e}")
                return False
        except Exception as e:
            print(f"[FAIL] Resume parser failed unexpectedly: {e}")
            return False
        finally:
            os.unlink(temp_file)

    except Exception as e:
        print(f"[FAIL] Resume parser test setup failed: {e}")
        return False

def test_signal_detector():
    """Test signal detector basic functionality"""
    try:
        from signal_detector import detect_role

        sample_signals = {
            'SDE': {
                'skills': ['python', 'algorithms'],
                'projects': ['web app'],
                'weight': 1.0
            },
            'Product Manager': {
                'skills': ['product', 'leadership'],
                'projects': ['product launch'],
                'weight': 1.0
            }
        }

        resume_text = "I am a software engineer with experience in python and algorithms."

        result = detect_role(resume_text, sample_signals)

        if 'role' in result and 'confidence' in result:
            print("[PASS] Signal detector works correctly")
            return True
        else:
            print("[FAIL] Signal detector returned invalid result")
            return False
    except Exception as e:
        print(f"[FAIL] Signal detector test failed: {e}")
        return False

def test_analyzer():
    """Test analyzer basic functionality"""
    try:
        from analyzer import get_disfluency_rate, get_consistency

        # Test disfluency rate
        text1 = "Hello world"
        rate1 = get_disfluency_rate(text1)
        assert rate1 == 0.0

        text2 = "um uh like you know foo bar baz qux quux"
        rate2 = get_disfluency_rate(text2)
        assert rate2 == 0.4  # 4 fillers / 10 words

        # Test consistency
        resume_text = "I have experience in Python and machine learning"
        answer_text = "I worked with Python and ML technologies"
        consistency = get_consistency(resume_text, answer_text)
        assert 0.0 <= consistency <= 1.0

        print("[PASS] Analyzer works correctly")
        return True
    except Exception as e:
        print(f"[FAIL] Analyzer test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running basic functionality tests for Sasha AI Interviewer...\n")

    tests = [
        test_imports,
        test_resume_parser,
        test_signal_detector,
        test_analyzer
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()  # Empty line between tests

    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("[PASS] All tests passed!")
        return 0
    else:
        print("[FAIL] Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())