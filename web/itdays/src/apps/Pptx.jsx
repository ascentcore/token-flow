import { useState, useEffect } from 'react';
// import 'react-tabs/style/react-tabs.css';

export default ({ images }) => {
    const [index, setIndex] = useState(0);

    const advance = () => {
        setIndex((prevState) =>
            prevState < images.length - 1 ? prevState + 1 : 0
        );
    };

    const handler = (e) => {
        if (e.key === 'ArrowRight' || e.key === ' ') {
            advance();
        } else if (e.key === 'ArrowLeft') {
            setIndex((prevState) => (prevState > 0 ? prevState - 1 : 0));
        }
    };

    useEffect(() => {
        window.addEventListener('keyup', handler, false);
        return () => window.removeEventListener('keyup', handler, false);
    }, []);

    return (
        <div className="image-slide" onClick={advance}>
            <div className="image-container">
                {images[index].indexOf('mp4') === -1 ? (
                    <img src={images[index]} alt="Slide" />
                ) : (
                    <video width="600" height="600" controls>
                        <source src={images[index]} type="video/mp4" />
                    </video>
                )}
            </div>
        </div>
    );
};
