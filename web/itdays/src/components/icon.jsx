export default (props) => {
  return (
    <span className="icon" onClick={props.onClick} style={props.style}>
      <img src={props.icon} alt="Logo" />
      <span className="title">{props.name}</span>
    </span>
  );
};
