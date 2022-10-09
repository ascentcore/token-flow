import * as ReactDOM from 'react-dom/client';
import Icon from '../components/icon';
import TextIcon from '/public/txt.png';

const texts = [
  { title: 'test.txt', text: 'This is a test text' },
  {
    title: 'mechanical_engine.txt',
    text: `An engine or motor is a machine designed to convert one or more forms of energy into mechanical energy. Available energy sources include potential energy (e.g. energy of the Earth's gravitational field as exploited in hydroelectric power generation), heat energy (e.g. geothermal), chemical energy, electric potential and nuclear energy (from nuclear fission or nuclear fusion). Many of these processes generate heat as an intermediate energy form, so heat engines have special importance. Some natural processes, such as atmospheric convection cells convert environmental heat into motion (e.g. in the form of rising air currents). Mechanical energy is of particular importance in transportation, but also plays a role in many industrial processes such as cutting, grinding, crushing, and mixing.`,
  },
];

export default (props) => {
  const openGeneric = props.openGeneric;

  return (
    <div className="folder">
      {texts.map((text) => (
        <Icon
          key={text.title}
          name={text.title}
          icon={TextIcon}
          onClick={openGeneric(
            text.title,
            TextIcon,
            <textarea style={{ width: '100%', height: '100%' }}>
              {text.text}
            </textarea>,
            800,
            700
          )}
        ></Icon>
      ))}
    </div>
  );
};
