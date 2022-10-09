import { useState } from 'react';

export default (props) => {

  const {closeWindow, createContext} = props;

  const [name, setName] = useState('new_context');
  const [initialWeight, setInitialWeight] = useState(0.2);
  const [weightIncrease, setWeightIncrease] = useState(0.2);
  const [tempDecrease, setTempDecrease] = useState(0.2);

  const createNewContext = () =>{
    const body = {
      name,
      initial_weight: initialWeight,
      weight_increase: weightIncrease,
      temp_decrease: tempDecrease 
    }

    createContext(body)
    closeWindow()
  }

  return (
    <div className="context-body">
      <h3>Create new context</h3>
      <div
        style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 10 }}
      >
        <div className="controls">
          <span className="label">Name:</span>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </div>
        <div className="controls">
          <span className="label">Initial Weight:</span>
          <input
            type="range"
            value={initialWeight}
            min={0}
            max={1}
            step={0.01}
            onChange={(e) => {
              setInitialWeight(parseFloat(e.target.value));
            }}
          />
          <span style={{ width: 50 }}>
            {Math.floor(initialWeight * 100) / 100}
          </span>
        </div>
        <div className="controls">
          <span className="label">Weight Increase:</span>
          <input
            type="range"
            value={weightIncrease}
            min={0}
            max={1}
            step={0.01}
            onChange={(e) => {
              setWeightIncrease(parseFloat(e.target.value));
            }}
          />
          <span style={{ width: 50 }}>
            {Math.floor(weightIncrease * 100) / 100}
          </span>
        </div>
        <div className="controls">
          <span className="label">Memory:</span>
          <input
            type="range"
            value={tempDecrease}
            min={0}
            max={1}
            step={0.01}
            onChange={(e) => {
              setTempDecrease(parseFloat(e.target.value));
            }}
          />
          <span style={{ width: 50 }}>
            {Math.floor(tempDecrease * 100) / 100}
          </span>
        </div>
      </div>
      <div style={{ textAlign: 'right', paddingTop: 20 }}>
        <button onClick={createNewContext}>Create Context</button>
      </div>
    </div>
  );
};
