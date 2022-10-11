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
import NewContextIcon from '/public/new_context.png';

import Contexts from './apps/Contexts';
import Context from './apps/Context';
import Exercise from './apps/Exercise';
import Vocabulary from './apps/Vocabulary';
import { registerListener, triggerEvent, unregisterListener } from './events';
import CheatSheets from './apps/CheatSheets';
import CreateContext from './apps/CreateContext';
function Button() {
  return <button>Openzz</button>;
}

function App() {
  const resetStimuli = (context) => () => {
    axios
      .post(`http://localhost:8081/reset_stimuli/${context}`)
      .then((response) => {
        const { data } = response;
        triggerEvent(context, data);
      })
      .catch((err) => {
        console.log(err);
      });
  };

  const stimulate = (context, token) => {
    axios
      .post(
        `http://localhost:8081/stimulate${
          context != null ? `/${context}` : ''
        }`,
        { text: token }
      )
      .then((response) => {
        const { data } = response;
        if (context) {
          triggerEvent(context, data);
        } else {
          triggerEvent('global', data);
        }
      })
      .catch((err) => {
        console.log(err);
      });
  };

  const createContext = (body) => {
    axios.post(`http://localhost:8081/create_context`, body);
  };

  function callBackendForText(context, sendValue, stimulate) {
    if (sendValue !== '') {
      axios
        .post(
          `http://localhost:8081/${
            stimulate ? `stimulate/${context}` : `add_text/${context}`
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

  const openContext = (context) => () => {
    setTimeout(() => {
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
        <Context
          context={context}
          callBackendForText={callBackendForText}
          resetStimuli={resetStimuli}
          stimulate={stimulate}
        />
      );
    }, 100);
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
        unregisterListener(callback);
      },
    });

    const callback = (data) => {
      box.setTitle(`Vocabulary (${data})`);
    };

    registerListener('vocabulary', callback);

    const root = ReactDOM.createRoot(box.body);
    root.render(<Vocabulary stimulate={stimulate} />);
  };

  const openNewContext = () => {
    const box = new WinBox({
      x: 'center',
      y: 'center',
      width: 550,
      height: 400,
      title: `New Context`,
      icon: NewContextIcon,
      border: 4,
      onclose: () => {
        root.unmount();
      },
    });

    const root = ReactDOM.createRoot(box.body);
    root.render(
      <CreateContext
        createContext={createContext}
        closeWindow={() => {
          box.close();
        }}
      />
    );
  };

  const openGeneric =
    (title, icon, component, width = 600, height = 350) =>
    () => {
      setTimeout(() => {
        const box = new WinBox({
          x: 'center',
          y: 'center',
          width,
          height,
          title: title,
          icon: icon,
          border: 4,
          onclose: () => {
            root.unmount();
          },
        });

        const root = ReactDOM.createRoot(box.body);
        root.render(component);
      }, 100);
    };

  // openContext('context1')()

  return (
    <div className="App">
      <Icon name="My Computer" icon={ComputerIcon}></Icon>
      <Icon
        name="Profile"
        icon={ProfileIcon}
        onClick={openGeneric('Profile', ProfileIcon, <Profile />, 600, 650)}
      ></Icon>
      <Icon
        name="Exercise"
        icon={ExerciseIcon}
        onClick={openGeneric('Exercise', ExerciseIcon, <Exercise />, 500, 600)}
      ></Icon>
      <Icon
        name="Vocabulary"
        icon={VocabularyIcon}
        onClick={openVocabulary}
      ></Icon>
      <Icon
        name="New Context"
        icon={NewContextIcon}
        onClick={openNewContext}
      ></Icon>
      <Icon
        name="Contexts"
        icon={FolderIcon}
        onClick={openGeneric(
          'Contexts',
          FolderIcon,
          <Contexts openContext={openContext} />
        )}
      ></Icon>
      <Icon
        name="Cheat Sheets"
        icon={FolderIcon}
        onClick={openGeneric(
          'Cheat Sheets',
          FolderIcon,
          <CheatSheets openGeneric={openGeneric} />
        )}
      ></Icon>
      <Icon name="Trash" icon={TrashIcon}></Icon>
    </div>
  );
}

export default App;
