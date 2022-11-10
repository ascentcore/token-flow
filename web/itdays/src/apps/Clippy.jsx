import { useEffect } from 'react';
import { useState } from 'react';
import { registerListener } from '../events';
import Clippy from '/public/clippy.png';

const messages = [
    'Looks like you are trying to do a presentation. Introduce yourself.',
    'With a high impostor syndrome',
    "Do the exercise! You'll find out who are the geeks.",
    'Is there something useful that we can get out of this?',
    'Looks... kinda dumb. Like somebody I know.',
    'We want demo, we want demo, we want demo!',
    'Uuuu, sarcasm! Can you do that?',
    'Somebody setup too many printers in its days. Good job.',
    'What about the sausage?',
    'Not that stupid after all ... unlike you!',
    'Is that a Simpsons reference? Booo, you are old!',
    [
        'Hi!',
        'Do you have 5 minutes?',
        'Hello',
        'Dud',
        'Dude',
        'I need help',
        'I have a question',
        'What are we eating today?',
        'Can you please give me feedback',
        'I need this by the end of the day!',
        'I\'m stuck!',
        'Can we skip the meeting today?',
        'U there?'
    ],
];

export default (props) => {
    const [message, setMessage] = useState(false);
    const [idx, setIndex] = useState(-1);
    const [showMessage, setShowMessage] = useState(false);

    useEffect(() => {
        registerListener('keyUp', (e) => {
            if (e === '§') {
                setShowMessage((val) => {
                    if (!val) {
                        setIndex((idx) => idx + 1);
                    }
                    return true;
                });
            }
        });

        registerListener('tabChange', (e) => {
            setShowMessage((val) => {
                console.log(e);
                if (e === 'profile-1') {
                    setIndex(1);
                    return true;
                } else if (e === 'exercise-1') {
                    setIndex(7);
                    return true;
                } else if (e === 'exercise-2') {
                    setIndex(10);
                    return true;
                }
                return false;
            });
        });

        registerListener('closed', (e) => {
            setShowMessage((val) => {
                console.log(e);
                if (e === 'Profile') {
                    setIndex(2);
                    return true;
                } else if (e === 'Exercise') {
                    setIndex(3);
                    return true;
                } else if (e === 'PowerPixel: Motivation') {
                    setIndex(5);
                    return true;
                } else if (e === 'PowerPixel: Working Together') {
                    setIndex(8);
                    return true;
                } else if (e === 'PowerPixel: Comparison') {
                    setIndex(9);
                    return true;
                }
                return false;
            });
        });

        registerListener('pptx', (e) => {
            setShowMessage((val) => {
                console.log(e);
                if (e === 'motivation-slide-3') {
                    setIndex(4);
                    return true;
                } else if (e === 'phonecalls-slide-3') {
                    setIndex(6);
                    return true;
                } else if (e === 'motivation-slide-6') {
                    setIndex(11);
                    return true;
                }
                return false;
            });
        });

        registerListener('talk', (e) => {
            setShowMessage((val) => {
                setMessage(e);
                return true;
            });
        });
    }, []);

    useEffect(() => {
        if (idx < messages.length) {
            setMessage(messages[idx]);
        }
    }, [idx]);

    if (message && showMessage)
        return (
            <div className="clippy" onClick={() => setShowMessage(false)}>
                <div class="messages">
                    {typeof message === 'string' ? (
                        <div className="message">{message}</div>
                    ) : (
                        message.map((msg) => (
                            <div className="message">{msg}</div>
                        ))
                    )}
                </div>
                <img src={Clippy} />
            </div>
        );
};
