from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from auth import get_current_user
from database import get_db
from models import User, UserBehavior
from schemas import BehaviorCreate, BehaviorOut, BehaviorUpdate

router = APIRouter(prefix="/behaviors", tags=["behaviors"])

@router.post("/", response_model=BehaviorOut, status_code=201)
def create_behavior(payload: BehaviorCreate, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    behavior = UserBehavior(**payload.model_dump())
    db.add(behavior)
    db.commit()
    db.refresh(behavior)
    return behavior

@router.get("/", response_model=List[BehaviorOut])
def list_behaviors(user_id: int = None, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    q = db.query(UserBehavior)
    if user_id:
        q = q.filter(UserBehavior.user_id == user_id)
    return q.order_by(UserBehavior.recorded_at.desc()).all()

@router.get("/{behavior_id}", response_model=BehaviorOut)
def get_behavior(behavior_id: int, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    b = db.query(UserBehavior).filter(UserBehavior.id == behavior_id).first()
    if not b:
        raise HTTPException(status_code=404, detail="Behavior record not found")
    return b

@router.put("/{behavior_id}", response_model=BehaviorOut)
def update_behavior(behavior_id: int, payload: BehaviorUpdate, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    b = db.query(UserBehavior).filter(UserBehavior.id == behavior_id).first()
    if not b:
        raise HTTPException(status_code=404, detail="Behavior record not found")
    for field, value in payload.model_dump(exclude_none=True).items():
        setattr(b, field, value)
    db.commit()
    db.refresh(b)
    return b

@router.delete("/{behavior_id}", status_code=204)
def delete_behavior(behavior_id: int, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    b = db.query(UserBehavior).filter(UserBehavior.id == behavior_id).first()
    if not b:
        raise HTTPException(status_code=404, detail="Behavior record not found")
    db.delete(b)
    db.commit()
