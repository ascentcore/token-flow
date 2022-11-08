import { useEffect } from 'react';
import { useState } from 'react';
import { registerListener } from '../events';
import Clippy from '/public/clippy.png';

const messages = [
    'Looks like you are trying to do a presentation. Introduce yourself.',
    'With a high impostor syndrome',
    "Do the exercise! You'll find out who are the geeks and who are the nerds.",
    'Tell them about the language models',
    'I am Clippy, your friendly assistant.',
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
                if (e === 'profile-1') {
                    setIndex(1);
                }
                return true;
            });
        });

        registerListener('pptx', (e) => {
            setShowMessage((val) => {
                console.log(e);
                // if (e === 'profile-1') {
                //     setIndex(1);
                // }
                // return true;
            });
        });
    }, []);

    useEffect(() => {
        console.log(idx);
        if (idx < messages.length) {
            setMessage(messages[idx]);
        }
    }, [idx]);

    if (message && showMessage)
        return (
            <div className="clippy" onClick={() => setShowMessage(false)}>
                <div className="message">{message}</div>
                <img src={Clippy} />
            </div>
        );
};
