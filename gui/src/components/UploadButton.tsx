import React from "react";
import { Button } from "@material-ui/core";

interface IUploadButtonProps {
  onClick: (event: any) => void;
  isActive: boolean;
}

const UploadButton: React.FC<IUploadButtonProps> = ({ onClick, isActive }) => {
  return (
    <div>
      <input
        accept="image/*"
        style={{ display: "none" }}
        id="icon-button-photo"
        onChange={onClick}
        type="file"
        disabled={!isActive}
      />
      <label htmlFor="icon-button-photo">
        <Button variant="contained" component="span" disabled={!isActive}>
          Upload Image
        </Button>
      </label>
    </div>
  );
};

export default UploadButton;
