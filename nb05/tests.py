import numpy as np
from numpy.testing import assert_almost_equal
from IPython.display import Markdown, display

# keeps track of the most recent test results
TEST_RESULTS = {
   'test_train_classifier' : {
        'description': '🎯 Tests: Implement the soft-margin classifier in general dimension',
        'status': '❓ untested'
    },
   'test_train_classifier_quad' : {
        'description': '🎯 Tests: Implement the soft-margin classifier with quadratic penalty',
        'status': '❓ untested'  
   }
}

def test_train_classifier(train_classifier, verbose=False):
    
    test_name = 'test_train_classifier'

    try :

        # hyperparameter for the test data
        C = 0.1

        # answers for the test data
        answers = [
            ([0.6830773529921718], -0.2185217761374115),
            ([-0.23372469360595777, 0.8500209604715415], -0.04130354492296835),
            ([-1.028005473858167, -0.2722065025123301,
            0.22276604553183557], 0.043229693481771164),
            ([
                -0.2523233127628585, -0.8043379752018387, 0.05866231524455665,
                -0.36535116749749047
            ], -0.07688342894923648),
            ([
                0.2715241418580811, 0.3864930229026365, 0.041752423483250645,
                0.29586749922211236, 0.8646554502597908
            ], -0.11740808507172118),
            ([
                -0.6001095841407434, 0.18488683159402186, 0.14078920824587154,
                -0.7269511741354757, 0.23951850971697417, 0.34352835941398757
            ], -0.01607971170138775),
            ([
                0.17477120903352597, -0.22601063725982337, -0.3148214803205057,
                -0.2699337088145239, 0.44392243941302634, -0.33123952121365735,
                -0.3538056062966607
            ], 0.06288600195057296)
        ]

        for i, (w_true, b_true) in enumerate(answers):

            # convert w_true into a numpy array
            w_true = np.array(w_true)

            # read the datafile
            data_filename = 'data/soft_margin_test_{}.npz'.format(i + 1)
            data = np.load(data_filename)

            # extract points and labels
            X, y = data['X'], data['y']

            # call student's classifier
            w, b = train_classifier(X, y, C)

            # print info and test
            if verbose:
                print('Test', i + 1)
                print('------')
                m, n = X.shape
                print('  {} points in {}D'.format(m, n))
                print()
                print('  w_true =', w_true)
                print('  w      =', w)
                assert_almost_equal(w, w_true)
                print()
                print('  b_true =', b_true)
                print('  b      =', b)
                assert_almost_equal(b, b_true)
                print()
                print('  PASSED')
                print()
    except Exception as e:
        TEST_RESULTS[test_name]['status'] = '❌ failed'
        print(f"❌ {test_name} failed: {e}")
        return
    TEST_RESULTS[test_name]['status'] = '✅ passed'
    print('✅ All tests passed!')

def test_train_classifier_quad(train_classifier_quad, verbose=False):

    test_name = 'test_train_classifier_quad'

    try :

        # hyperparameter for the test data
        C = 0.1

        # answers for the test data
        answers = [
            ([0.32533851395953417], -0.08646719746462925),
            ([-0.16670553714486364, 0.5034552900223377], -0.008797835597748514),
            ([-0.5714821569508671, -0.17558696237104554,
            0.1487830244679727], 0.009342523241439343),
            ([
                -0.16820166540832585, -0.553971386905195, 0.016189156220830784,
                -0.3138718205203185
            ], -0.03995885060199727),
            ([
                0.11656724459864966, 0.26268682974980556, -0.002095108258010457,
                0.13958346667452734, 0.5514022000446535
            ], -0.10283511905859569),
            ([
                -0.4697315521337579, 0.11652068745496669, 0.12136849945088767,
                -0.5438042798361219, 0.20152083750327335, 0.2350336142244464
            ], 0.032840282466452944),
            ([
                0.10587106346447647, -0.18438002497367256, -0.19143132321832493,
                -0.24896753820314582, 0.2807287163385554, -0.20548935039161917,
                -0.32475823594704734
            ], 0.08199571409290467)
        ]

        for i, (w_true, b_true) in enumerate(answers):

            # convert w_true into a numpy array
            w_true = np.array(w_true)

            # read the datafile
            data_filename = 'data/soft_margin_test_{}.npz'.format(i + 1)
            data = np.load(data_filename)

            # extract points and labels
            X, y = data['X'], data['y']

            # call student's classifier
            w, b = train_classifier_quad(X, y, C)

            # print info and test
            if verbose:
                print('Test', i + 1)
                print('------')
                m, n = X.shape
                print('  {} points in {}D'.format(m, n))
                print()
                print('  w_true =', w_true)
                print('  w      =', w)
                assert_almost_equal(w, w_true)
                print()
                print('  b_true =', b_true)
                print('  b      =', b)
                assert_almost_equal(b, b_true)
                print()
                print('  PASSED')
                print()
    except Exception as e:
        TEST_RESULTS[test_name]['status'] = '❌ failed'
        print(f"❌ {test_name} failed: {e}")
        return

    TEST_RESULTS[test_name]['status'] = '✅ passed'
    print('✅ All tests passed!')

def test_summary():
    md = "## Test Summary\n\n| Test Description | Status |\n|---|---|\n"
    for _, info in TEST_RESULTS.items():
        md += f"| {info['description']} | {info['status']} |\n"
    display(Markdown(md))