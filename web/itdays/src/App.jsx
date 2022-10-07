import './App.css';
import 'winbox/dist/winbox.bundle.min.js';
import * as ReactDOM from 'react-dom/client';
import Profile from './apps/Profile';
import Icon from './components/icon';

import axios from 'axios';
import ReactDOMServer from 'react-dom/server';

import ProfileIcon from '/public/profile_icon.png';
import VocabularyIcon from '/public/vocabulary.png';
import ExerciseIcon from '/public/exercise.png';

import ComputerIcon from '/public/computer.png';
import TrashIcon from '/public/trash.png';
import FolderIcon from '/public/folder.png';
import ContextIcon from '/public/context.png';

import Contexts from './apps/Contexts';
import Context from './apps/Context';
import Exercise from './apps/Exercise';
import Vocabulary from './apps/Vocabulary';
import { registerListener, triggerEvent, unregisterListener } from './events';
function Button() {
  return <button>Openzz</button>;
}

function App() {
  function callBackendForText(context, sendValue, stimulate) {
    if (sendValue !== '') {
      axios
        .post(
          `http://localhost:8081/${
            stimulate ? `stimulate/${context}` : 'add_text'
          }`,
          {
            text: sendValue,
          }
        )
        .then((response) => {
          const { data } = response;
          if (!stimulate) {
            triggerEvent('global', data);
          } else {
            triggerEvent(context, data);
          }
        });
    }
  }

  const openSlides =
    (Title, Component, Icon, width = 600, height = 650) =>
    () => {
      const box = new WinBox({
        x: 'center',
        y: 'center',
        width,
        height,
        title: Title,
        icon: Icon,
        border: 4,
        onclose: () => {
          root.unmount();
        },
      });
      const root = ReactDOM.createRoot(box.body);
      root.render(Component);
    };

  const openContext = (context) => () => {
    const box = new WinBox({
      x: 'center',
      y: 'center',
      width: 800,
      height: 700,
      title: `${context} context`,
      icon: ContextIcon,
      border: 4,

      onclose: () => {
        root.unmount();
      },
    });

    const root = ReactDOM.createRoot(box.body);
    root.render(
      <Context context={context} callBackendForText={callBackendForText} />
    );
  };

  const openVocabulary = () => {
    const box = new WinBox({
      x: 'center',
      y: 'center',
      width: 300,
      height: 700,
      title: `Vocabulary`,
      icon: VocabularyIcon,
      border: 4,
      onclose: () => {
        root.unmount();
        unregisterListener(callback)
      },
    });

    const callback = (data) => {
      box.title = `Vocabulary (${data})`;
    }

    registerListener('vocabulary', callback);

    const root = ReactDOM.createRoot(box.body);
    root.render(<Vocabulary />);
  };

  const openContexts = () => {
    const box = new WinBox({
      x: 'center',
      y: 'center',
      width: 600,
      height: 350,
      title: 'Contexts',
      icon: FolderIcon,
      border: 4,

      onclose: () => {
        root.unmount();
      },
    });

    const root = ReactDOM.createRoot(box.body);
    root.render(<Contexts openContext={openContext} />);
  };

  // openContext('context1')();

  return (
    <div className="App">
      <Icon name="My Computer" icon={ComputerIcon}></Icon>
      <Icon
        name="Profile"
        icon={ProfileIcon}
        onClick={openSlides('Profile', <Profile />, ProfileIcon)}
      ></Icon>
      <Icon
        name="Exercise"
        icon={ExerciseIcon}
        onClick={openSlides('Exercise', <Exercise />, ExerciseIcon, 500)}
      ></Icon>
      <Icon
        name="Vocabulary"
        icon={VocabularyIcon}
        onClick={openVocabulary}
      ></Icon>
      <Icon name="Contexts" icon={FolderIcon} onClick={openContexts}></Icon>
      <Icon name="Trash" icon={TrashIcon}></Icon>
    </div>
  );
}

export default App;
