import * as ReactDOM from 'react-dom/client';
import Icon from '../components/icon';
import TextIcon from '/public/txt.png';

const texts = [
  {
    title: 'software_engine.txt',
    text: `A software engine is a computer program, or part of a computer program, that serves as the core foundation for a larger piece of software. This term is often used in game development, in which it typically refers to either a graphics engine or a game engine around which the rest of a video game is developed. While the term can also be used in other areas of software development, its particular meaning can be more nebulous in those instances. A software engine can be developed by a company that is using it, or may be developed by another company and then licensed to other developers. When used in the general context of computer software development, a software engine typically refers to the core elements of a particular program. This usually does not include features such as the user interface (UI) and numerous art assets added to the core engine itself. For an operating system (OS), for example, the software engine might be the source code that establishes file hierarchy, input and output methods, and how the OS communicates with other software and hardware. The exact contents of such an engine can vary from program to program, however. In computer and console game development, a software engine typically refers to either a game’s graphics engine or the overall game engine. The graphics engine for a game is typically the software used to properly render out the graphics seen by players. This often uses art assets created in other programs, which are then ported into the graphics engine for use during game play. The use of a software engine for the graphics of a game can make rendering much easier, and may also simplify the process of ensuring software and hardware compatibility.`,
  },
  {
    title: 'mechanical_engine.txt',
    text: `The engine of a vehicle is the part that converts the energy of the fuel into mechanical energy, and produces the power which makes the vehicle move. He got into the driving seat and started the engine. Automotive Engines are generally classified according to following different categories: Internal combustion (IC) and External Combustion (EC) Type of fuel: Petrol, Diesel, Gas, Bio / Alternative Fuels. Number of strokes – Two stroke Petrol, Two-Stroke Diesel, Four Stroke Petrol / Four Stroke Diesel. The engine is the vehicle's main source of power. The engine uses fuel and burns it to produce mechanical power. The heat produced by the combustion is used to create pressure which is then used to drive a mechanical device. The engine is a lot like the brain of a car. It holds all the power necessary to help your car function. And without it, your car would be nothing. But there are multiple car engine types out there on the road. An engine, or motor, is a machine used to change energy into a movement that can be used. The energy can be in any form. Common forms of energy used in engines are electricity, chemical (such as petrol or diesel), or heat. When a chemical is used to produce the energy it is known as fuel.`,
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
