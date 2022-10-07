export default (props) => {
  return (
    <span className="icon" onClick={props.onClick}>
      <img src={props.icon} alt="Logo" />
      <span className="title">{props.name}</span>
    </span>
  );
};
